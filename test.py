import networkx as nx
import matplotlib.pyplot as plt
import itertools as it
import string
import sys
import cmath as cmath
import math as math
from networkx.algorithms import isomorphism


# creates a new kgraph of v vertices and ka colors
def new_kgraph(v,ka):
    nodes = string.ascii_lowercase[:v]
    G=nx.MultiDiGraph(k=ka)
    G.add_nodes_from(nodes)
    return G

#example 1: valid 2graph with parallel edges and two self loops
def ex1():
    nodes = ['v','w']
    G=nx.MultiDiGraph(k=2)
    G.add_nodes_from(nodes);
    # G.add_edge('v','w', k=k[0])
    # G.add_edge('v','w', k=k[0])
    G.add_edge('v','w', k=1)
    G.add_edge('v','w', k=1)
    G.add_edge('v','w', k=1)
    G.add_edge('v','v', k=2) 
    G.add_edge('w','w', k=2)
    # G.add_edge('w','u', k=1)
    # G.add_edge('u','v', k=1)
    return G

#example 2: valid 2graph with no self loops
def ex2():
    nodes = ['u','v','w','x','y']
    G=nx.MultiDiGraph(k=2)
    G.add_nodes_from(nodes);
    # G.add_edge('v','w', k=k[0])
    # G.add_edge('v','w', k=k[0])
    G.add_edge('y','w', k=1)
    G.add_edge('y','w', k=1)
    G.add_edge('x','w', k=1)
    G.add_edge('x','w', k=1)
    G.add_edge('v','u', k=1)
    G.add_edge('v','u', k=1)
    G.add_edge('x','v', k=2)
    G.add_edge('x','v', k=2)
    G.add_edge('y','v', k=2)
    G.add_edge('y','v', k=2)
    G.add_edge('w','u', k=2)
    G.add_edge('w','u', k=2)
    return G

#example 3: valid 2graph with no self loops or parallel edges
def ex3():
    nodes = ['alpha','beta','gamma','delta']
    G=nx.MultiDiGraph(k=2)
    G.add_nodes_from(nodes);
    G.add_edge('alpha','beta', k=1)
    G.add_edge('gamma','delta', k=1)
    G.add_edge('beta','delta', k=2)
    G.add_edge('alpha','gamma', k=2)
    return G

#example 4: NOT valid 3graph with no self loops or parallel edges
def ex4():
    G = ex3()
    G.graph['k']=3
    G.add_node('epsilon')
    G.add_edge('delta','epsilon', k=3)
    return G

    #example 5: valid 3graph with self loops
def ex5():
    G = ex3()
    G.graph['k']=3
    G.add_node('epsilon')
    G.add_edge('epsilon','alpha', k=3)
    G.add_edge('epsilon','beta', k=1)
    G.add_edge('epsilon','gamma', k=2)
    G.add_edge('alpha','alpha', k=3)
    G.add_edge('beta','beta', k=3)
    G.add_edge('gamma','gamma', k=3)
    G.add_edge('delta','delta', k=3)
    return G

def ex6():
    nodes = ['b','a']
    G=nx.MultiDiGraph(k=2)
    G.add_nodes_from(nodes);
    G.add_edge('a','b', k=2)
    G.add_edge('a','a', k=2)
    G.add_edge('b','b', k=1)
    return G

# Not a valid kgraph
def ex7():
    nodes = ['a','b']
    G=nx.MultiDiGraph(k=3)
    G.add_nodes_from(nodes);
    G.add_edge('a','b', k=1)
    G.add_edge('a','a', k=2)
    G.add_edge('b','b', k=3)
    return G

def ex8():
    G = new_kgraph(3,3)
    # nodes = ['a','b','c']
    # G=nx.MultiDiGraph(k=3)
    # G.add_nodes_from(nodes);
    G.add_edge('a','a', k=1)
    G.add_edge('b','b', k=1)
    G.add_edge('c','a', k=1)
    G.add_edge('b','c', k=2)
    G.add_edge('b','a', k=3)
    return G

def ex9():
    nodes = ['b','a']
    G=nx.MultiDiGraph(k=2)
    G.add_nodes_from(nodes);
    G.add_edge('a','b', k=1)
    G.add_edge('a','b', k=2)
    # G.add_edge('a','a', k=1)
    # G.add_edge('b','b', k=1)
    # G.add_edge('b','b', k=2)
    return G



def draw_kgraph(G,suppress_draw=False,save_as=None):
    plt.clf()
    k = G.graph['k']
    #edges = [ (u,v) for u,v,edata in G.edges(data=True) if edata == 'r']
    pos=nx.spring_layout(G)
    #Draw Nodes
    nx.draw_networkx_nodes(G,pos,
        nodelist=G.nodes(),
        node_color='k',
        node_size=500,
        alpha=0.8)
    # l = {}
    # l['v']=r'$v$'
    # l['w']=r'$w$'
    #print labels
    #
    #draws the 
    # draw_networkx_edges(G,pos,width=1.0,alpha=0.5)
    for e in range(1,k+1):
        edgelist = []
        for x in [ ((u,v),edata['k']) for u,v,edata in G.edges(data=True) if 'k' in edata]:
            #print x
            if x[1] == e:
                edgelist.insert(0,x[0])
        #creates the list of colors for drawing edges
        colors = ['black','red','blue','green','yellow']
        #print edgelists
        draw_networkx_edges(G,pos,edgelist=edgelist,width=3,alpha=0.5,edge_color=colors[e])
    #draws the graph labels
    l = labels(G.nodes())
    nx.draw_networkx_labels(G,pos,labels=l,font_size=20,font_color='w')

    plt.axis('off')
    if not(suppress_draw):
        plt.show()
    else:
        if(save_as != None):
            plt.savefig(save_as)
    #A=nx.to_agraph(G)        # convert to a graphviz graph
    #A.layout()            # neato layout
    #A.draw("k5.png")       # write postscript in k5.ps with neato layout
    #nx.write_dot(G,'multi.dot')
    return

#labels the nodes in a kgraph
def labels(A):
    d = dict()
    for x in A:
        if len(x) == 1:
            d[str(x)]=r'${}$'.format(x)
        else:
            d[str(x)]=r'$\{}$'.format(x)
    return d

#quickly determines if a kgraph G is not a valid kgraph
#if false, then G is definately not a valid kgraph
# if true, G may or may not be a valid kgraph
def quick_invalid(G):
    k = G.graph['k']
    colors = [False] * k
    edges = nx.get_edge_attributes(G,'k')
    # print type(edges)
    for v in edges.values():
        colors[v-1]=True
    for i in colors:
        if (i == False):
            return False
    return True



#returns true/false if G abides by the factorization property
def valid_kgraph(G,suppress_warnings=False):
    k = G.graph['k']
    #a multidigraph is a valid kgraph if it abides by the factorization property
    #this property states that for every path from node A to B that follows a red-blue edge
    #there must exist a path from A to B that follows a blue-red edge.
    cycles = list(nx.simple_cycles(G))
    has_cycle = len(cycles) != 0
    if not suppress_warnings:
        print "Known cycles: {0}".format(cycles)
        # print cycle_permutation(cycles)
        if(has_cycle):
            print "WARNING: Graph is cyclical, and this methods accuracy is questionable"
    #initialize knownpathlist; a set of tuples containing start/end nodes, and their path permutation
    #tuples here designate valid paths that are known to exist in the graph
    knownpathlist = []
    #initialize neededpathlist; a set of tuples containing start/end nodes, and their path permutation
    #tuples here must also exist in knownpathlist if G is a valid kgraph
    neededpathlist = []
    v = 2 if (len(G.nodes()) < 3) else 3
    #Get every node
    for p in build_paths(G,v):
        #adds the new known path and its path permutation to the knownpathlist
        knownpathlist.append(p)
        for x in build_needed_paths(p):
            neededpathlist.append(x)
                # if(a == 'b' and b == 'a'):
                #     print "PANIC OUTPUT"
                #     print "f = {0}".format(f)
                #     print "y = {0}".format(y)
                #     print "p = {0}".format(p)
                #     print "pl = {0}".format(pl)
                #     print "q = {0}".format(q)
                #     print "a = {0}".format(a)
                #     print "b = {0}".format(b)
    #after iterating through every node and finding every path from every other node,
    #we verify that every tuple in neededpathlist is also in knownpathlist
    #If there is a tuple in neededpathlist not in knownpathlist, it will attempt to look for its complex path.
    #if it can't find it, then G is not a valid Kgraph
    # print "known: {0}".format(knownpathlist)
    # print "needed: {0}".format(neededpathlist)
    # print "---"
    for x in neededpathlist:
        if x not in knownpathlist:
            #a needed path is not known. maybe it needs to cycle?
            if(has_complex_perm_path(G,x)):
                knownpathlist.append(x)
                continue
            print "path not known! {0}".format(x)
            return False
    return True

#returns an array containing all 1graphs through kgraphs of v vertices with no parallel edges
#this method generates a big list: consider the set of all 2graphs with 2 vertices
#that is a total of 64 different graphs!
#because this implmenetation is naive, many graphs are either not connected, not of k different colors,
#or even valid kgraphs
def all_kgraphs_naive(V,K):
    graphlist = []
    kp = K+1
    # This generates the set of all nonparallel kgraphs by generating the set of all numbers from 0 to 
    # (K+1)^(V^2)
    #for each edge, it either 
    for i in range(((kp)**(V*V))):
        # print "---"
        # print "i = {0}".format(i)
        # The algorithm used is actually the same to convert an integer into a (V^2) digit base-K number
        # pretty cool, huh?
        # clones i
        j = i
        # resets index
        index = 0
        # digit is the array of digits holding our graph info
        digit = [0] * (V*V)
        # print "j = {0}".format(j)
        while ( j != 0 ):
            remainder = j % (kp) # assume K > 1
            j = j / (kp) #integer division
            digit[index] = remainder
            index += 1
        G = new_kgraph(V,K)
        #for every edge
        print digit
        for m in range(V):
            
            # print m
            for n in range(V):
                # print n
                #gets the index of the edge
                index = (m*V)+n
                # if the set digit is zero, do not add the edge
                # print "{0} {1}".format(index,digit[index])
                if(digit[index] != 0):
                    
                #otherwise, add the edge according to stored 
                    G.add_edge(string.ascii_lowercase[m],string.ascii_lowercase[n],k=digit[index])
        if not nx.is_weakly_connected(G):
            continue
        if not quick_invalid(G):
            continue
        graphlist.append(G)
    return graphlist

# Given a list of graphs and a desired k, trims the list by removing all 
def trim_graphs(graphlist, k):
    trimmed = []
    # print type(graphlist)
    for g in graphlist:
        # print type(g)
        if valid_kgraph(g,True):
            trimmed.append(g)
    return trimmed

# returns true if kgraph g is invalid
def invalid(g):
    k = g.graph['k']
    if not nx.is_weakly_connected(g):
        return True
    if not quick_invalid(g):
        return True
    if not valid_kgraph(g,True):
        return True
    return False

# currently not working
def trim_isomorphic(graphlist):
    trimmed = []
    for i in graphlist:
        for j in trimmed:
            GM = isomorphism.GraphMatcher(i,j)
            if GM.is_isomorphic():
                break
        trimmed.append(i)
        print trimmed
    return trimmed






# Returns true or false if there exists a complex path from vertices a to b 
# that follows the permutation path stored in c
# Note that complex paths include loops
def has_complex_perm_path(G,perm_path):
    #get edge attributes
    edges = nx.get_edge_attributes(G,'k')
    #start node
    a = perm_path[0]
    # print "a = {0}".format(a)
    #end node
    b = perm_path[1]
    # print "b = {0}".format(b)
    #path permutation
    c = perm_path[2]
    # print "perm path = {0}".format(c)
    #desired edgnetwe color to follow
    color = c[0]
    #foreach edge attached to our start node
    for i in nx.edges(G,a):
        #next is the vertice the edge points to
        next = i[1]
        #HACK: edge attributes need the id number of the edge to get the attributes
        #Because this is not included by default with nx.edges(G,a), there can be bugs
        #down the line with differently colored parallel edges not seen by this method
        i += (0,)
        #d is the color of the next vertice
        d = edges[i]
        #if c is the color we are looking for
        if(d == color):
            #if this is the only path we needed to look for
            if(len(c)==1):
                # we have found the last path. we better be at our destination.
                if(next != b):
                    return False
                # otherwise next=b and we have found a complex path from a to b following c
                # percolate true
                return True
            #we recurse, looking for a complex path from next to b, with a split permutation path
            if(has_complex_perm_path(G,(next,b,c[1:]))):
                return True
    # a complex path could not be found. 
    # percolate false
    return False
list()




#returns the path permutation
#a path permutation is an ordered list of what colors were traveled across
#G is the graph, path is the path traveled, and k is the number of the kgraph
def path_permutation(G,path,parallel=0):
    k = G.graph['k']
    #initialize edges, the set of all edge combinations in the 
    edges = nx.get_edge_attributes(G,'k')
    # print edges
    #initialize perms, the set of all edge combinations
    perms = []
    # print path
    #for every edge in the path
    for p in zip(path,path[1:]):
        #reset the list of perms
        # perms = []
        # print "p is {0}".format(p)
        # print "parallel is {0}".format(parallel)
        passes = parallel
        while True:
            try:
                l = G.get_edge_data(p[0],p[1],passes)
                color = l['k']
            except TypeError:
                passes -= 1
                continue
            perms.append(color)
            break
    return perms

#returns the shape of a permutation path
def shape(path,g):
    k = g.graph['k']
    shape = {}
    for i in range(k):
        shape[i] = 0;
    for i in path:
        shape[i-1] += 1
    return tuple(j for i,j in shape.items())

#given a list of simple cycles, this function returns an expanded list containing duplicate cycles that also start from
#other nodes in each cycle
def cycle_permutation(cycles):
    returnlist = []
    for x in cycles:
        l = len(x)
        if(l == 1):
            returnlist.append(x)
            continue
        clone = x[:]
        for i in range(l):
            q = clone[:]
            elem = q.pop(0)
            q.append(elem)
            returnlist.append(q)
            clone = q
    return returnlist





def draw_networkx_edges(G, pos,
                        edgelist=None,
                        width=1.0,
                        edge_color='k',
                        style='solid',
                        alpha=1.0,
                        edge_cmap=None,
                        edge_vmin=None,
                        edge_vmax=None,
                        ax=None,
                        arrows=True,
                        arrowstyle='thick',
                        label=None,
                        **kwds):
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.cbook as cb
        import matplotlib.patches as patches
        from matplotlib.colors import colorConverter, Colormap
        from matplotlib.collections import LineCollection
        from matplotlib.path import Path
        import numpy
    except ImportError:
        raise ImportError("Matplotlib required for draw()")
    except RuntimeError:
        print("Matplotlib unable to open display")
        raise
    # print "drawing_edges"

    if ax is None:
        ax = plt.gca()

    if edgelist is None:
        edgelist = G.edges()

    if not edgelist or len(edgelist) == 0:  # no edges!
        return None

    # set edge positions
    edge_pos = numpy.asarray([(pos[e[0]], pos[e[1]]) for e in edgelist])
    # for e in edge_pos:
    #   print e
    if not cb.iterable(width):
        lw = (width,)
    else:
        lw = width

    if not cb.is_string_like(edge_color) \
           and cb.iterable(edge_color) \
           and len(edge_color) == len(edge_pos):
        if numpy.alltrue([cb.is_string_like(c)
                         for c in edge_color]):
            # (should check ALL elements)
            # list of color letters such as ['k','r','k',...]
            edge_colors = tuple([colorConverter.to_rgba(c, alpha)
                                 for c in edge_color])
        elif numpy.alltrue([not cb.is_string_like(c)
                           for c in edge_color]):
            # If color specs are given as (rgb) or (rgba) tuples, we're OK
            if numpy.alltrue([cb.iterable(c) and len(c) in (3, 4)
                             for c in edge_color]):
                edge_colors = tuple(edge_color)
            else:
                # numbers (which are going to be mapped with a colormap)
                edge_colors = None
        else:
            raise ValueError('edge_color must consist of either color names or numbers')
    else:
        if cb.is_string_like(edge_color) or len(edge_color) == 1:
            edge_colors = (colorConverter.to_rgba(edge_color, alpha), )
        else:
            raise ValueError('edge_color must be a single color or list of exactly m colors where m is the number or edges')

    edge_collection = LineCollection(edge_pos,
                                     colors=edge_colors,
                                     linewidths=lw,
                                     antialiaseds=(1,),
                                     linestyle=style,
                                     transOffset = ax.transData,
                                     )

    # print type(edge_collection)

    edge_collection.set_zorder(1)  # edges go behind nodes
    edge_collection.set_label(label)
    ax.add_collection(edge_collection)

    # Note: there was a bug in mpl regarding the handling of alpha values for
    # each line in a LineCollection.  It was fixed in matplotlib in r7184 and
    # r7189 (June 6 2009).  We should then not set the alpha value globally,
    # since the user can instead provide per-edge alphas now.  Only set it
    # globally if provided as a scalar.
    if cb.is_numlike(alpha):
        edge_collection.set_alpha(alpha)

    if edge_colors is None:
        if edge_cmap is not None:
            assert(isinstance(edge_cmap, Colormap))
        edge_collection.set_array(numpy.asarray(edge_color))
        edge_collection.set_cmap(edge_cmap)
        if edge_vmin is not None or edge_vmax is not None:
            edge_collection.set_clim(edge_vmin, edge_vmax)
        else:
            edge_collection.autoscale()

    arrow_collection = None

    if G.is_directed() and arrows:

        # a directed graph hack-fix
        # draws arrows at each
        # waiting for someone else to implement arrows that will work
        arrow_colors = edge_colors
        a_pos = []
        p = .1  # make arrows 10% of total length
        angle = 2.7 #angle for arrows
        for src, dst in edge_pos:
            x1, y1 = src
            x2, y2 = dst
            dx = x2-x1   # x offset
            dy = y2-y1   # y offset
            d = numpy.sqrt(float(dx**2 + dy**2))  # length of edge
            theta = numpy.arctan2(dy, dx)
            if d == 0:   # source and target at same position
                continue
            if dx == 0:  # vertical edge
                xa = x2
                ya = dy+y1
            if dy == 0:  # horizontal edge
                ya = y2
                xa = dx+x1
            else:
                # xa = p*d*numpy.cos(theta)+x1
                # ya = p*d*numpy.sin(theta)+y1
                #corrects the endpoints to better draw 
                x2 -= .04 * numpy.cos(theta)
                y2 -= .04 * numpy.sin(theta)
                lx1 = p*d*numpy.cos(theta+angle)+(x2)
                lx2 = p*d*numpy.cos(theta-angle)+(x2)
                ly1 = p*d*numpy.sin(theta+angle)+(y2)
                ly2 = p*d*numpy.sin(theta-angle)+(y2)

            
            a_pos.append(((lx1, ly1), (x2, y2)))
            a_pos.append(((lx2, ly2), (x2, y2)))

        arrow_collection = LineCollection(a_pos,
                                colors=arrow_colors,
                                linewidths=[1*ww for ww in lw],
                                antialiaseds=(1,),
                                transOffset = ax.transData,
                                )

        arrow_collection.set_zorder(1)  # edges go behind nodes
        arrow_collection.set_label(label)
        # print type(ax)
        ax.add_collection(arrow_collection)

    #drawing self loops

    d = 1
    c = 0.0707
    selfedges = []
    verts = [
    (0.1*d - 0.1*d, 0.0), # P0
    (c * d- 0.1*d, c * d), # P0
    (0.0- 0.1*d, 0.1 * d), # P0
    (-c * d- 0.1*d, c * d), # P0
    (-0.1 * d- 0.1*d, 0.0), # P0
    (-c * d- 0.1*d, -c * d), # P0
    (0.0- 0.1*d, -0.1 * d), # P0
    (c * d - 0.1*d, -c * d), # P0
    (0.1*d - 0.1*d, 0.0)
    ]
    # print verts

    codes = [Path.MOVETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO,
    ]

    for e in edge_pos:
        if(numpy.array_equal(e[0],e[1])):
            nodes = verts[:]
            for i in range(len(nodes)):
                nodes[i] += e[0]
            # print nodes
            path = Path(nodes, codes)
            patch = patches.PathPatch(path, color=None, facecolor=None,edgecolor=edge_colors[0],fill=False, lw=4)
            ax.add_patch(patch)


    # update view
    minx = numpy.amin(numpy.ravel(edge_pos[:, :, 0]))
    maxx = numpy.amax(numpy.ravel(edge_pos[:, :, 0]))
    miny = numpy.amin(numpy.ravel(edge_pos[:, :, 1]))
    maxy = numpy.amax(numpy.ravel(edge_pos[:, :, 1]))

    w = maxx-minx
    h = maxy-miny
    padx,  pady = 0.05*w, 0.05*h
    corners = (minx-padx, miny-pady), (maxx+padx, maxy+pady)
    # print ax
    ax.update_datalim(corners)
    ax.autoscale_view()

#    if arrow_collection:

    return edge_collection


def paths_length_x(G,a,b,x=3,cycles=None):
    if(cycles==None):
        cycles = list(nx.simple_cycles(G))
    minlength = x-1
    paths = list(nx.all_simple_paths(G, a,b))
    if(len(cycles)==0):
        return paths
    pathset = paths[:]
    newpathset = []
    #repeats this method minlength times
    for i in range(minlength):
        expandedpaths = []
        for c in cycles:
            c0 = c[0]
            for i in pathset:
                for j in range(len(i)):
                    if(i[j] == c0):
                        ii = i[:]
                        ii[j:j] = c
                        # print "original path is {0}, index is {1}, cycle is {2}, newpath is {3}".format(i,j,c,ii)
                        expandedpaths.append(ii)
                        if(len(ii) < x+1):
                            newpathset.append(ii)
                        break
        paths.extend(expandedpaths)
        # print "expandedpaths = {0}".format(expandedpaths)
        pathset = newpathset[:]
        newpathset = []
    # print "paths: {0}".format(paths)
    return paths




# Kgraphs are multidigraphs, are not acyclic, and are multicolored.
# They have few guarentees of shape, size or performance.
# Consider the task of finding all paths from a to b of no specified length
# because a kgraph is not a DAG, there can an infinite number of paths from a to b
# if there is a cycle between a and b, BFS would become indefinitely big
# This version of BFS returns the list of all paths of finite length from a not extending past b
def bfs(graph, a,b,length=3):
    # maintain a queue of paths
    complete = []
    queue = []
    # push the first path into the queue
    for i in graph.neighbors(a):
        queue.append([a,i])
        complete.append([a,i])
    while queue:
        # get the first path from the queue
        path = queue.pop(0)
        if(len(path)==length+1):
            # complete.append(path)
            continue
        # get the last node from the path
        node = path[-1]
        # path found
        if node == b:
            # complete.append(path)
            continue
        # enumerate all adjacent nodes, construct a new path and push it into the queue
        for adjacent in graph.neighbors(node):
            # print "adj = {0}".format(adjacent)
            new_path = list(path)
            new_path.append(adjacent)
            complete.append(new_path)
            queue.append(new_path)

    return complete

# builds the set of known paths of maximum length v
# returns a list of unique tuples containing start node, end node and path permutation
def build_paths(G,v=3):
    paths = list()
    for a in G.nodes():
        for b in G.nodes():
            for y in bfs(G,a,b,v):
                # print "YYY {0}".format(y)
                # gets the ending node of the bfs path
                f = y[-1]
                # parallel = 1
                parallel = G.number_of_edges(a,f)
                # print "a / f / parallel is {0} {1} {2}".format(a,f,parallel)
                for c in range(parallel):
                    p = path_permutation(G,y,c)
                    pl = (a,f,tuple(p),tuple(y))
                    # print pl
                    #adds the new known path and its path permutation to the knownpathlist
                    # print "Pl {0}".format(pl)
                    paths.append(pl)
    return list(set(paths))

# builds a list of needed paths from p such that G would be a valid kgraph

def build_needed_paths(p):
    needed = list()
    # creates q, the list of path permutations from a to b that must also 
    # be in G if G is a valid kgraph
    q = list(set(it.permutations(p[2])))
    # print "p = {0}".format(p)
    # print "q = {0}".format(q)
    for x in q:
        ql = (p[0],p[1],x)
        needed.append(ql)
    return list(set(needed))

# builds the set of equivalence classes
def build_equivalence_classes(G):
    v = 2 if (len(G.nodes()) < 3) else 3
    equiv = {}
    paths = build_paths(G,v)
    for x in paths:
        # print "path is {0}".format(x)
        _shape = shape(x[2],G)
        # print "shape is {0}".format(_shape)
        key = (x[0],x[1],_shape)
        if key not in equiv.keys():
            equiv[key] = list()
        # append x to our equivalence class
        # hacky way to account for equivalence classes as a result of parallel edges.
        # All it does is ads the same path c times
        # where c is the number of paralel paths
        # This will have to be changed if account for something other than edge color.
        for c in range(G.number_of_edges(x[0],x[1])):
            equiv[key].append(x)
    return equiv

def print_equiv_classes(equiv):
    for k in equiv.keys():
        print "{0}".format(k)
        for v in equiv[k]:
            print "\t{0}".format(v)

# apends to an equiv class a new value designating the assigned cocycle value
# in this case it assigns the value 1
def assign_trivial_2ocycle(equiv):
    appended = {}
    for k in equiv.keys():
        appended[k] = list()
        for v in equiv[k]:
            b = v + ((1 + 0j),)
            appended[k].append(b[:])
    return appended

# apends to an equiv class a new value designating the assigned cocycle value
# in this case it assigns
def assign_complex_2cocycle(equiv):
    appended = {}
    twopi = 2* math.pi
    for k in equiv.keys():
        appended[k] = list()
        # gets the number of relations in the equivalence class
        count = max(len(equiv[k]),1)
        # print count
        i = 0
        theta = (twopi / count)
        for v in equiv[k]:
            c = unit_circle_complex(theta*i)
            b = v + (c,)
            appended[k].append(b[:])
            i += 1
    return appended

# gets the value of the complex number on the unit circle at a certain
def unit_circle_complex(theta, sigfigs = 6):
    # return 1 in the complex plane if theta is 0
    if(theta == 0):
        return (1 + 0j)
    r = round(math.cos(theta),sigfigs)
    i = round(math.sin(theta),sigfigs)
    return complex(r, i)

# takes a kgraph and runs a gauntlet of tests.
# Begins by determining if it is a valid kgraph.
# if not, it stops execution.
# it then builds it's equivalence classes and assigns arbitray complex 2cocycles
def kgraph_gauntlet(G):
    valid = valid_kgraph(G, True)
    print "G is a valid kgraph: {0}".format(valid)
    if not valid:
        return
    equiv = build_equivalence_classes(G)
    cocycles = assign_complex_2cocycle(equiv)
    print_equiv_classes(cocycles)
    draw_kgraph(G,True,"gauntlet.png")


#This is the immediate code that gets run
G = ex5()
kgraph_gauntlet(G)
# print "G is a valid kgraph: {0}".format(valid_kgraph(G))
# print "G is is_weakly_connected: {0}".format(valid_kgraph(G))
# print bfs(G,'a','b',3)
# print "has complex perm path = {0}".format(has_complex_perm_path(G, ('b', 'a', (1, 2))))
# print nx.get_edge_attributes(G,'k')
# draw_kgraph(G,True,"fdsa.png")
# equiv_classes = build_equivalence_classes(G)
# print "equivalence classes = \n{0}".format(equiv_classes)
# print "\n\n"
# print "equivalence class keys = \n{0}".format(equiv_classes.keys())
# equiv_classes = assign_complex_2cocycle(equiv_classes)
# print_equiv_classes(equiv_classes)

# # for i in nx.all_simple_paths(G, 'v','w'): 
# #     print i
# # for i in nx.simple_cycles(G):
# #     print i
# print "G is an invalid kgraph: {0}".format(invalid(G,2))

# graphset = all_kgraphs_naive(2,2)
# print "how many naive graphs: {0}".format(len(graphset))
# trimmedset = trim_graphs(graphset,3)
# print "how many trimmed graphs: {0}".format(len(trimmedset))
# i = 0
# for g in trimmedset:
#     i += 1
#     # print "G is a valid kgraph: {0}".format(valid_kgraph(g))
#     # print g.edges()
#     # print g.nodes()
#     draw_kgraph(g,True,"2_2/{0}.png".format(i))

