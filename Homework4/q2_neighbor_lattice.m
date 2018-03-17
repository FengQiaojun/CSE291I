function neigs = q2_neighbor_lattice(index,size)
    neigs = [];
    if index-1>=1 && mod(index-1,size)~=0
        neigs = [neigs,index-1];
    end
    if index+1<=size^2 && mod(index+1,size)~=1
        neigs = [neigs,index+1];
    end
    if index-size>=1
        neigs = [neigs,index-size];
    end
    if index+size<=size^2
        neigs = [neigs,index+size];
    end
end