import numpy as np
from solidspy import solids_GUI
import matplotlib.pyplot as plt
import solidspy.preprocesor as pre
import solidspy.postprocesor as pos
import solidspy.assemutil as ass
import solidspy.solutil as sol
import cv2

size = 64

def topo_to_tab(topology):
    tab = np.zeros(size*size, dtype = int)
    for i in range(size):
        for j in range(size):
            tab[i+j*64] = 1 if topology[i,j] < 127 else 0
    return tab

def create_files(topology, BC_conf, load_position, load_x_value, load_y_value, directory):
    tab = topo_to_tab(topology)
    nodes_of_topo = []
    # nodes.txt file
    # Boundary conditions of nodes
    BC_node = np.zeros(((size+1)**2, 2))
    for elem in BC_conf:
        list_nodes = elem[0]
        type_bc = elem[1]
        for n in list_nodes:
            if type_bc == 1 or type_bc == 3:
                BC_node[n-1, 0] = -1
            if type_bc == 2 or type_bc == 3:
                BC_node[n-1, 1] = -1
    # Creating the file
    f = open(f"{directory}/nodes.txt", "w")
    for node in range(1, (size+1)**2 + 1):
    # Coordinates of nodes
        x = node//(size+1)
        r = node % (size+1)
        if r != 0:
            y = (size+1) - r
        else:
            x -= 1
            y = 0

        f.write(f"{node - 1}  {x:.2f}  {y:.2f}  {BC_node[node-1,0]:.0f}  {BC_node[node-1,1]:.0f}" + "\n")
    f.close()
    
    # eles.txt file
    f = open(f"{directory}/eles.txt", "w")
    num_elem = 0
    for node in range(1, (size+1)**2 + 1):
        if node % (size+1) != 0 and node < (size+1)**2-size:
            f.write(f"{num_elem}  1  {tab[num_elem]}  {node - 1}  {node - 1 + 1}  {node - 1 + (size+2)}  {node - 1 + (size+1)}" + "\n")
            num_elem += 1
            if num_elem < size**2 and tab[num_elem] == 1:
                nodes_of_topo.append(node-1)
                nodes_of_topo.append(node)
                nodes_of_topo.append(node+size+1)
                nodes_of_topo.append(node+size)
    f.close()
    
    # mater.txt file
    f = open(f"{directory}/mater.txt", "w")
    f.write("1e-3  0.3" + "\n")
    f.write("1.0  0.3")
    f.close()
    
    # loads.txt file
    f = open(f"{directory}/loads.txt", "w")
    for i, pos in enumerate(load_position):
        f.write(f"{pos - 1}  {load_x_value[i]:.1f}  {load_y_value[i]:.1f}" + "\n")
    f.close()
    
    return np.unique(np.array(nodes_of_topo))

def resize(arr):
    res = np.empty((64,64))
    for i in range(64):
        for j in range(64):
            res[i,j] = arr[i,j]+arr[i+1,j]+arr[i,j+1]+arr[i+1,j+1]
    return res*0.25

def mysolidspy(path):
    compute_strains = True
    plot_contours = False

    nodes, mats, elements, loads = pre.readin(folder=path)

    # Pre-processing
    DME , IBC , neq = ass.DME(nodes, elements)
    print("Number of nodes: {}".format(nodes.shape[0]))
    print("Number of elements: {}".format(elements.shape[0]))
    print("Number of equations: {}".format(neq))

    # System assembly
    KG = ass.assembler(elements, mats, nodes, neq, DME)
    RHSG = ass.loadasem(loads, IBC, neq)

    # System solution
    UG = sol.static_sol(KG, RHSG)
    if not(np.allclose(KG.dot(UG)/KG.max(), RHSG/KG.max())):
        print("The system is not in equilibrium!")

    # Post-processing
    UC = pos.complete_disp(IBC, nodes, UG)
    E_nodes, S_nodes = None, None
    if compute_strains:
        E_nodes, S_nodes = pos.strain_nodes(nodes, elements, mats, UC)
    if plot_contours:
        pos.fields_plot(elements, nodes, UC, E_nodes=E_nodes, S_nodes=S_nodes)

    return (UC, E_nodes, S_nodes) if compute_strains else UC


def fem_compliance_i(topology,dict_array,i):
    print(f"analyzing sample {i+1}")
    BC_conf = dict_array[i]['BC_conf']
    load_position = dict_array[i]['load_nodes']
    load_x_value = dict_array[i]['x_loads']
    load_y_value = dict_array[i]['y_loads']
    nodes_topo = create_files(np.squeeze(topology[i]), BC_conf, load_position, load_x_value, load_y_value, "./fem_files/")
    
    UC_in, E_nodes_in, S_nodes_in = mysolidspy("./fem_files/")

    return np.sum(np.multiply(E_nodes_in, S_nodes_in))

def check_load(load_coord, sample):
    coord_x = int(load_coord[0] * 63)
    coord_y = int(63 - load_coord[1] * 63)
    return sample[coord_y][coord_x] >= 127

def compute_vf(image, bandw = True):
    _, bw_img = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
    if not bandw :
        bw_img = np.mean(bw_img, axis = 2)
    return 1 - np.count_nonzero(bw_img)/(64*64)

def check_floating_material(image):
    ret, im = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    im = im.astype(np.uint8)
    sa = cv2.connectedComponents(im)
    return not sa[0] == 2

def compute_deflection(num_samples, summaries, gen):
    res = np.empty(num_samples)
    for i in range(num_samples):
        res[i] = fem_compliance_i(gen, summaries, i)
    return res

def compute_load(num_samples, summaries, gen):
    res = np.empty(num_samples, dtype = int)
    for i in range(num_samples):
        res[i] = check_load(summaries[i]["load_coord"][0], gen[i])
    return res

def compute_vf_error(num_samples, summaries, gen):
    res = np.empty(num_samples)
    for i in range(num_samples):
        res[i] = np.abs(compute_vf(gen[i])-summaries[i]["VF"])/summaries[i]["VF"]
    return res

def compute_fm(num_samples, gen):
    res = np.empty(num_samples, dtype = int)
    for i in range(num_samples):
        res[i] = check_floating_material(gen[i])
    return res

def analysis(num_samples, summaries, gen):
    d = compute_deflection(num_samples, summaries, gen)
    l = compute_load(num_samples, summaries, gen)
    vf = compute_vf_error(num_samples, summaries, gen)
    fm = compute_fm(num_samples, gen)
    return d, l, vf, fm

def order_list(num_samples, num_folder):
    lst = np.empty(num_samples, dtype = int)
    s = []
    for i in range(num_folder):
        s.append(str(i))     
    s.sort()
         
    for i in range(num_samples):
        lst[i] = int(s[i])
    
    return lst

def re_order_tab(num_samples, num_folder, tab):
    lst = order_list(num_samples, num_folder)
    return np.array([tab[i] for i in lst])

def pre_process_summaries(dict_array, lst = None):
    for i in range(dict_array.size):
        load_nodes_i = np.empty(dict_array[i]['load_coord'].shape[0])
    for j,coord in enumerate(dict_array[i]['load_coord']):
        node = int(round(64*coord[0])*65+round(64*(1.0 - coord[1])))
        if node < 0:
            node = 0
        load_nodes_i[j] = node + 1
    dict_array[i]['load_nodes'] = load_nodes_i.astype(int)

    if lst is not None:
        dict_arrays = [dict_array[i] for i in lst]
    else:
        dict_arrays = dict_array
    return dict_arrays

def topodiff_analysis(num_samples, num_folder, summaries, gen_dir):

    lst = order_list(num_samples, num_folder)
    summaries = pre_process_summaries(summaries, lst)

    gen = np.load(gen_dir + f"samples_{num_samples}x64x64x1.npz")["arr_0"]
    gen[gen < 127] = 0.
    gen[gen >= 128] = 255.
    return analysis(num_samples, summaries, gen)

def simp_analysis(num_samples, summaries, topy_path):
    summaries = pre_process_summaries(summaries)
    gen = np.load(topy_path)
    gen[gen < 0.5] = 0
    gen[gen >= 0.5] = 1
    gen = gen.astype(bool)
    gen = np.invert(gen)
    gen = gen.astype(float)
    gen = gen * 255
    gen = gen.reshape((num_samples, 64, 64, 1))
    return analysis(num_samples, summaries, gen)

def print_results(tab):
    print(f"Average CE: {tab[0].mean()}\nAverage VFE: {tab[2].mean()}\nProportion of LD: {tab[1].mean()}\nProportion of FM: {tab[3].mean()}")