import os
import pandas as pd 
import re
import errno
import theoretical_functions
import matplotlib.pyplot as plt
import numpy as np
import pdb

def distance(x1,x2):
    return (x1[0]-x2[0])**2+(x1[1]-x2[1])**2+(x1[2]-x2[2])**2

def PBC_done_right(x1, border):
    xl=border[0][1]-border[0][0]
    yl=border[1][1]-border[1][0]
    zl=border[2][1]-border[2][0]
    dx=x1[0]
    dy=x1[1]
    dz=x1[2]
    eps=0.00001
    #if(dx==30.00):
    #    pdb.set_trace()
    while(dx>=xl/2-eps):
        dx=dx-xl
    while(dx<-xl/2):
        dx=dx+xl
    while(dy>=yl/2-eps):
        dy=dy-yl
    while(dy<-yl/2):
        dy=dy+yl
    while(dz>=zl/2-eps):
        dz=dz-zl
    while(dz<-zl/2):
        dz=dz+zl
    return [dx,dy,dz]


def distance_PBC(x1,x2, border):
    x2start=x2
    xl=border[0][1]-border[0][0]
    yl=border[1][1]-border[1][0]
    zl=border[2][1]-border[2][0]
    dx=x2[0]-x1[0]
    dy=x2[1]-x1[1]
    dz=x2[2]-x1[2]
    while(dx>xl/2):
        dx=dx-xl
    while(dx<-xl/2):
        dx=dx+xl
    while(dy>yl/2):
        dy=dy-yl
    while(dy<-yl/2):
        dy=dy+yl
    while(dz>zl/2):
        dz=dz-zl
    while(dz<-zl/2):
        dz=dz+zl
    return np.sqrt(dx**2+dy**2+dz**2)

def radial_distribution_done_right(input_file,id1,id2,n_bins=100,max_length=1000,max_distance=None):
    f=open(input_file)
    number_found=0
    atom_number=0
    temp={}
    borders=[]
    distances=[]
    step_count=0
    box_size_found=0
    while (number_found==0 or box_size_found==0):
        if('ITEM: NUMBER OF ATOMS' in f.readline()):
            number_found=1
            atom_number=int(f.readline())
        if('ITEM: BOX BOUNDS' in f.readline()):
            box_size_found=1
            borders.append([float(s) for s in(f.readline().split(' '))])
            borders.append([float(s) for s in(f.readline().split(' '))])
            borders.append([float(s) for s in(f.readline().split(' '))])
    counter=0
    max_distance=np.sqrt(3)*np.abs((borders[0][0]-borders[0][1]))*0.6 if max_distance==None else max_distance
    bin_size=max_distance/(2*n_bins)
    distance_bins=np.arange(0,max_distance,bin_size)
    distance_histo=np.zeros(n_bins*2)
    if id1==id2:
        temp=[]
        for line in f:
            temp=[]
            tempid=[]
            if('ITEM: ATOMS' in line):
                for i in range(atom_number):
                    line=[float(s) for s in(f.readline().split(' '))]
                    if line[int(1)]==id1:
                        temp.append(line[2:5])
                        tempid.append(line[0])
                    
                randomOrder=list(range(len(temp)))
                np.random.shuffle(randomOrder)
                temp=[temp[i] for i in randomOrder]
                tempid=[tempid[i] for i in randomOrder]
                for i in range(int(len(temp))):
                    for j in range(i+1,len(temp)):
                        distance12=distance_PBC(temp[i],temp[j],borders)
                        if distance12<max_distance:
                            index=np.digitize(distance12,distance_bins, right=True)
                            distance_histo[index]=distance_histo[index]+1
                            counter=counter+1
                            if counter>max_length:
                                return(rdf_from_histo(distance_histo,max_distance,n_bins))
    else:    
        for line in f:
            temp[id1]=[]
            temp[id2]=[]
            if('ITEM: ATOMS' in line):

                for i in range(atom_number):
                    line=[float(s) for s in(f.readline().split(' '))]
                    if(int(line[1])==id1 or int(line[1])==id2):
                        temp[int(line[1])].append(line[2:5])
                for i in range(len(temp[id1])):
                    for j in range(len(temp[id2])):
                        distance12=distance_PBC(temp[id1][i],temp[id2][j],borders)
                        if distance12<max_distance:
                            index=np.digitize(distance12,distance_bins)
                            distance_histo[index]=distance_histo[index]+1
                            counter=counter+1
                            if counter>max_length:
                                return(rdf_from_histo(distance_histo,max_distance,n_bins))
    distance_histo=distance_histo[0:int(n_bins)]
    x=np.linspace(0,max_distance/2,n_bins)[0:-1]
    for i in range(len(distance_histo)):
        shellVolume=4*np.pi*((((i+1)*bin_size))**3-(i*bin_size)**3)/3
        distance_histo[i]=distance_histo[i]/shellVolume
    normalization=np.mean(distance_histo[int(len(distance_histo)/2):int(2*len(distance_histo)/3)])
    distance_histo=distance_histo[0:-1]
    distance_histo=[i/normalization for i in distance_histo]
    return x,distance_histo

def rdf_from_histo(histo,max_distance,n_bins):
    bin_size=max_distance/(2*n_bins)
    histo=histo[0:int(n_bins)]
    x=np.linspace(0,max_distance/2,n_bins)[0:-1]
    for i in range(len(histo)):
        shellVolume=4*np.pi*((((i+1)*bin_size))**3-(i*bin_size)**3)/3
        histo[i]=histo[i]/shellVolume
    normalization=np.mean(histo[int(len(histo)/2):int(2*len(histo)/3)])
    histo=histo[0:-1]
    histo=[i/normalization for i in histo]
    return x,histo


def radial_distribution(input_file,id1,id2,max_length=10000000):
    f=open(input_file)
    number_found=0
    atom_number=0
    temp={}
    borders=[]
    distances=[]
    step_count=0
    box_size_found=0
    while (number_found==0 or box_size_found==0):
        if('ITEM: NUMBER OF ATOMS' in f.readline()):
            number_found=1
            atom_number=int(f.readline())
        if('ITEM: BOX BOUNDS' in f.readline()):
            box_size_found=1
            borders.append([float(s) for s in(f.readline().split(' '))])
            borders.append([float(s) for s in(f.readline().split(' '))])
            borders.append([float(s) for s in(f.readline().split(' '))])
    counter=0
    max_distance=np.sqrt(3)*np.abs((borders[0][0]-borders[0][1]))
    if id1==id2:
        temp=[]
        for line in f:
            temp=[]
            tempid=[]
            if('ITEM: ATOMS' in line):
                counter=counter+1

                for i in range(atom_number):
                    line=[float(s) for s in(f.readline().split(' '))]
                    if line[int(1)]==id1:
                        temp.append(line[2:5])
                        tempid.append(line[0])
                    
                randomOrder=list(range(len(temp)))
                np.random.shuffle(randomOrder)
                temp=[temp[i] for i in randomOrder]
                tempid=[tempid[i] for i in randomOrder]
                for i in range(int(len(temp))):
                    for j in range(i+1,len(temp)):
                        distance12=distance_PBC(temp[i],temp[j],borders)
                        if distance12<max_distance:
                            distances.append(distance12)
                        if(len(distances)>max_length):
                            return distances
        return distances
    
    for line in f:
        temp[id1]=[]
        temp[id2]=[]
        if('ITEM: ATOMS' in line):
            counter=counter+1

            for i in range(atom_number):
                line=[float(s) for s in(f.readline().split(' '))]
                if(int(line[1])==id1 or int(line[1])==id2):
                    temp[int(line[1])].append(line[2:5])
            for i in range(len(temp[id1])):
                for j in range(len(temp[id2])):
                    distance12=distance_PBC(temp[id1][i],temp[id2][j],borders)
                    if distance12<max_distance:
                        distances.append(distance12)
                    if(len(distances)>max_length):
                        return distances
    return distances

def radial_distribution_polymer(input_file,id1,threshold):
    f=open(input_file)
    number_found=0
    atom_number=0
    temp={}
    borders=[]
    distances=[]
    step_count=0
    box_size_found=0
    while (number_found==0 or box_size_found==0):
        if('ITEM: NUMBER OF ATOMS' in f.readline()):
            number_found=1
            atom_number=int(f.readline())
        if('ITEM: BOX BOUNDS' in f.readline()):
            box_size_found=1
            borders.append([float(s) for s in(f.readline().split(' '))])
            borders.append([float(s) for s in(f.readline().split(' '))])
            borders.append([float(s) for s in(f.readline().split(' '))])
    counter=0
    max_distance=np.sqrt(3)*np.abs((borders[0][0]-borders[0][1]))
    temp=[]
    for line in f:
        temp=[]
        tempid=[]
        if('ITEM: ATOMS' in line):
            counter=counter+1
            for i in range(atom_number):
                line=[float(s) for s in(f.readline().split(' '))]
                if line[1]==id1:
                    temp.append(line[2:5])
                    tempid.append(line[0])
            for i in range(int(len(temp)-1)):
                for j in range(i+1,len(temp)):
                    if np.abs(tempid[i]-tempid[j])>threshold:
                        distance12=distance_PBC(temp[i],temp[j],borders)
                        if distance12<max_distance:
                            distances.append(distance12)
            if(len(distances)>10000000):
                return distances
    return distances


def gyration_radius(input_file,t, tstart=0):
    f=open(input_file)
    number_found=0
    atom_number=0
    xMean=0.0
    yMean=0.0
    zMean=0.0
    meanGyrRadius=0.0
    step_count=0
    box_size_found=0
    x_start=0.0
    x_stop=0.0
    toReturn=[]
    while (number_found==0 or box_size_found==0):
        if('ITEM: NUMBER OF ATOMS' in f.readline()):
            number_found=1
            atom_number=int(f.readline())
        if('ITEM: BOX BOUNDS' in f.readline()):
            box_size_found=1
            size=[float(s) for s in(f.readline().split(' '))]
            x_start=size[0]
            x_stop = size[1]
    next_line=f.readline()    
    while (step_count<tstart and next_line != ''):
        if('ITEM: ATOMS' in next_line):
            step_count=step_count+1
        next_line=f.readline()
    stepcount=0
    while (step_count<t and next_line != ''):
        
        if('ITEM: ATOMS' in next_line):
            xmean=0.0
            ymean=0.0
            zmean=0.0
            xgyr=0.0
            ygyr=0.0
            zgyr=0.0
            temp=[]
            
            for i in range(atom_number):
                temp.append([float(s) for s in(f.readline().split(' '))][2:])
                xmean= xmean+temp[i][0]
                ymean= ymean+temp[i][1]
                zmean= zmean+temp[i][2]
            xmean=xmean/atom_number
            ymean=ymean/atom_number
            zmean=zmean/atom_number
            for i in range(atom_number):
                xgyr= xgyr+(xmean-temp[i][0])**2
                ygyr= ygyr+(ymean-temp[i][1])**2
                zgyr= zgyr+(zmean-temp[i][2])**2                
            toReturn.append(np.sqrt((xgyr+ygyr+zgyr)/atom_number))
            step_count=step_count+1
        next_line=f.readline()
    f.close()
    return toReturn

def gyration_radius_sigma(input_file,t, tstart=0, block_size=100):
    f=open(input_file)
    number_found=0
    atom_number=0
    xMean=0.0
    yMean=0.0
    zMean=0.0
    meanGyrRadius=0.0
    step_count=0
    box_size_found=0
    x_start=0.0
    x_stop=0.0
    toReturn=[]
    while (number_found==0 or box_size_found==0):
        if('ITEM: NUMBER OF ATOMS' in f.readline()):
            number_found=1
            atom_number=int(f.readline())
        if('ITEM: BOX BOUNDS' in f.readline()):
            box_size_found=1
            size=[float(s) for s in(f.readline().split(' '))]
            x_start=size[0]
            x_stop = size[1]
    next_line=f.readline()    
    while (step_count<tstart and next_line != ''):
        if('ITEM: ATOMS' in next_line):
            step_count=step_count+1
        next_line=f.readline()
    step_count=0
    while (step_count<t and next_line != ''):
        
        if('ITEM: ATOMS' in next_line):
            xmean=0.0
            ymean=0.0
            zmean=0.0
            xgyr=0.0
            ygyr=0.0
            zgyr=0.0
            temp=[]
            
            for i in range(atom_number):
                temp.append([float(s) for s in(f.readline().split(' '))][2:])
                xmean= xmean+temp[i][0]
                ymean= ymean+temp[i][1]
                zmean= zmean+temp[i][2]
            xmean=xmean/atom_number
            ymean=ymean/atom_number
            zmean=zmean/atom_number
            for i in range(atom_number):
                xgyr= xgyr+(xmean-temp[i][0])**2
                ygyr= ygyr+(ymean-temp[i][1])**2
                zgyr= zgyr+(zmean-temp[i][2])**2                
            toReturn.append(np.sqrt((xgyr+ygyr+zgyr)/atom_number))
            step_count=step_count+1
        next_line=f.readline()
    random.shuffle(toReturn)
    expected_value=[]
    expected_value_squared=[]
    for i in range(int(len(toReturn)/block_size)):
        expected_value.append(np.mean(toReturn[i*block_size:(i+1)*block_size]))
        expected_value_squared.append(np.mean(toReturn[i*block_size:(i+1)*block_size])**2)
    f.close()
    return expected_value,expected_value_squared


def gyration_radius_verbose(input_file,timeFrame):
    f=open(input_file)
    number_found=0
    atom_number=0
    xMean=0.0
    yMean=0.0
    zMean=0.0
    meanGyrRadius=0.0
    step_count=0
    box_size_found=0
    x_start=0.0
    x_stop=0.0
    toReturn=[]
    while (number_found==0 or box_size_found==0):
        if('ITEM: NUMBER OF ATOMS' in f.readline()):
            number_found=1
            atom_number=int(f.readline())
        if('ITEM: BOX BOUNDS' in f.readline()):
            box_size_found=1
            size=[float(s) for s in(f.readline().split(' '))]
            x_start=size[0]
            x_stop = size[1]
    next_line=f.readline()    
    while (step_count<timeFrame and next_line != ''):
        if('ITEM: ATOMS' in next_line):
            step_count=step_count+1
        next_line=f.readline()
    while (next_line != ''):
        if(step_count==timeFrame):
            if('ITEM: ATOMS' in next_line):
                xmean=0.0
                ymean=0.0
                zmean=0.0
                xgyr=0.0
                ygyr=0.0
                zgyr=0.0
                temp=[]
                tempID=[]
                for i in range(atom_number):
                    temp.append([float(s) for s in(f.readline().split(' '))])
                    tempID.append(temp[i][0:2])
                    temp[i]=temp[i][2:]
                    xmean= xmean+temp[i][0]
                    ymean= ymean+temp[i][1]
                    zmean= zmean+temp[i][2]
                xmean=xmean/atom_number
                ymean=ymean/atom_number
                zmean=zmean/atom_number
                for i in range(atom_number):
                    xgyr= xgyr+(xmean-temp[i][0])**2
                    ygyr= ygyr+(ymean-temp[i][1])**2
                    zgyr= zgyr+(zmean-temp[i][2])**2                
                toReturn.append(np.sqrt((xgyr+ygyr+zgyr)/atom_number))
                step_count=step_count+1
                print(f'mean:           x:{xmean} y:{ymean} z:{zmean}')
                print(f'gyration_radius: x:{xgyr} y:{ygyr} z:{zgyr}')
                print(f'gyration_radius: {toReturn}')
                for i in range(atom_number):
                    print(tempID[i])
                    print(temp[i])
                    
                return 0
        next_line=f.readline()
    f.close()
    return toReturn

def end_to_end_distance(input_file,t,t_start=0):
    f=open(input_file)
    number_found=0
    atom_number=0
    
    xMean=0.0
    yMean=0.0
    zMean=0.0
    end_to_end_distances=[]
    step_count=0
    box_size_found=0
    mean_kuhn_length=0.0
    end_to_end_distance_x=0
    end_to_end_distance_y=0
    end_to_end_distance_z=0
    while (number_found==0 or box_size_found==0):
        if('ITEM: NUMBER OF ATOMS' in f.readline()):
            number_found=1
            atom_number=int(f.readline())
        if('ITEM: BOX BOUNDS' in f.readline()):
            box_size_found=1
            size=[float(s) for s in(f.readline().split(' '))]
            x_start=size[0]
            x_stop = size[1]
    
    for line in f:
        
        if('ITEM: ATOMS' in line):
            temp=[[]]*atom_number 
            for i in range(atom_number):
                next_line=[s for s in(f.readline().split(' '))]
                temp[int(next_line[0])-1]=[float(s) for s in next_line[2:]]
                
            step_count=step_count+1
            
            end_to_end_distance_x=(temp[0][0]-temp[-1][0])**2
            end_to_end_distance_y=(temp[0][1]-temp[-1][1])**2
            end_to_end_distance_z=(temp[0][2]-temp[-1][2])**2
            end_to_end_distances.append(np.sqrt(end_to_end_distance_x+end_to_end_distance_y+end_to_end_distance_z))
    return np.array(end_to_end_distances)


def swelling_parameter_sigma(input_file):
    f=open(input_file)
    number_found=0
    atom_number=0
    block_size=100
    step_count=0
    box_size_found=0
    mean_kuhn_length=0.0
    end_to_end_distance_=0
    end_to_end_distance_x=0
    end_to_end_distance_y=0
    end_to_end_distance_z=0
    positions=[]
    means=[]
    meansSquare=[]
    positions=[]
    while (number_found==0 or box_size_found==0):
        if('ITEM: NUMBER OF ATOMS' in f.readline()):
            number_found=1
            atom_number=int(f.readline())
        if('ITEM: BOX BOUNDS' in f.readline()):
            box_size_found=1
            size=[float(s) for s in(f.readline().split(' '))]
            x_start=size[0]
            x_stop = size[1]
    
    for line in f:
        temp=[]
        if('ITEM: ATOMS' in line):
            temp=[[]]*atom_number
            step_count=step_count+1
            for i in range(atom_number):
                    next_line=[s for s in(f.readline().split(' '))]
                    temp[int(next_line[0])-1]=[float(s) for s in next_line[2:]]
            positions.append(temp)
            
            
    f.close()
    random.shuffle(positions)
    n_blocks=(step_count//block_size)
    N=n_blocks*block_size
    i=0
    while (i<n_blocks-1):
        j=0
        end_to_end_distance_block_mean=0.0
        mean_kuhn_length=0.0
        offset=i*block_size
        while (j<block_size):
            kuhn_length=0.0
            for k in range(atom_number-1):
                deltax=(positions[offset+j+1][k][0]-positions[offset+j][k][0])**2
                deltay=(positions[offset+j+1][k][1]-positions[offset+j][k][1])**2
                deltaz=(positions[offset+j+1][k][2]-positions[offset+j][k][2])**2
                kuhn_length=kuhn_length+np.sqrt(deltax+deltay+deltaz)
            mean_kuhn_length=mean_kuhn_length+kuhn_length/(atom_number-1)
            step_count=step_count+1
            end_to_end_distance_x=(positions[offset+j+1][0][0]-positions[offset+j+1][-1][0])**2
            end_to_end_distance_y=(positions[offset+j+1][0][1]-positions[offset+j+1][-1][1])**2
            end_to_end_distance_z=(positions[offset+j+1][0][2]-positions[offset+j+1][-1][2])**2
            end_to_end_distance_block_mean=end_to_end_distance_block_mean+end_to_end_distance_x+end_to_end_distance_y+end_to_end_distance_z
            j=j+1
        end_to_end_distance_block_mean=end_to_end_distance_block_mean/block_size
        r0=(atom_number-1)*(mean_kuhn_length/block_size)**2    
        means.append((end_to_end_distance_block_mean/r0))
        meansSquare.append(((end_to_end_distance_block_mean/r0)**2))
        i=i+1
        
def rdf_from_distances(distance_distribution,bins=500):
    max_distance=3*max(distance_distribution)/5
    histo=(bins+1)*[0]
    bin_size=(max_distance)/bins
    minDistance=min(distance_distribution)
    for i in (distance_distribution):
        if(i<max_distance):
            histo[int(bins*i/max_distance)]=histo[int(bins*i/max_distance)]+1
    for i in range(len(histo)):
        shellVolume=4*np.pi*((((i+1)*bin_size))**3-(i*bin_size)**3)/3
        histo[i]=histo[i]/shellVolume
    x=np.linspace(0,max_distance,len(histo))[0:-1]
    normalization=np.mean(histo[int(len(histo)/2):int(2*len(histo)/3)])
    histo=histo[0:-1]
    histo=[i/normalization for i in histo]
    return x,histo

def gYDSMathematica(xs,etaTot,sizeRatio,molarFraction):
    nco=len(sizeRatio)
    session = WolframLanguageSession()
    session.evaluate(wl.SetDirectory("/home/eugenio/Desktop/tesi/hardSpherePotential/"))
    session.evaluate(wl.Needs("YDS`","./_mathematicaModule.m"))
    session.evaluate(wl.ResetDirectory())
    alpha=session.evaluate(wlexpr(f"ALFA[,{nco}, {{{NPoly/Ntot},{N/Ntot},{M/Ntot}}}, {{1,2/3,1/3}}]"))
    RDFMathematica = lambda r: session.evaluate(wlexpr(f"RDF[1,1,{r},100,1,0.49, {{1}}, {{1}},0.02912874461548902]"))



def kuhn_length_distribution(input_file):
    f=open(input_file)
    number_found=0
    atom_number=0
    step_count=0
    box_size_found=0
    mean_kuhn_lengths=0.0
    while (number_found==0 or box_size_found==0):
        if('ITEM: NUMBER OF ATOMS' in f.readline()):
            number_found=1
            atom_number=int(f.readline())
        if('ITEM: BOX BOUNDS' in f.readline()):
            box_size_found=1
            size=[float(s) for s in(f.readline().split(' '))]
            x_start=size[0]
            x_stop = size[1]
    kuhn_lengths=[]
    for line in f:
        
        temp=[[]]*atom_number
        if('ITEM: ATOMS' in line):
            for i in range(atom_number):
                    next_line=[s for s in(f.readline().split(' '))]
                    temp[int(next_line[0])-1]=[float(s) for s in next_line[2:]]
            for i in range(atom_number-1):
                deltax=(temp[i+1][0]-temp[i][0])**2
                deltay=(temp[i+1][1]-temp[i][1])**2
                deltaz=(temp[i+1][2]-temp[i][2])**2
                kuhn_lengths.append(np.sqrt(deltax+deltay+deltaz))
    return kuhn_lengths

def mean_squared_displacement(input_file, atom_ID):
    f=open(input_file)
    number_found=0
    atom_number=0
    step_count=0
    box_size_found=0
    mean_kuhn_lengths=0.0
    flag=0
    while (number_found==0 or box_size_found==0):
        if('ITEM: NUMBER OF ATOMS' in f.readline()):
            number_found=1
            atom_number=int(f.readline())
        if('ITEM: BOX BOUNDS' in f.readline()):
            box_size_found=1
            size=[float(s) for s in(f.readline().split(' '))]
            x_start=size[0]
            x_stop = size[1]
    kuhn_lengths=[]
    starting_pos=np.full([atom_number,3],np.nan)
    #starting_pos[:]=np.nan
    msds=[]
    counter=0
    for line in f: 
        temp=np.full([atom_number,3],np.nan)
        #temp[:]=np.nan
        if('ITEM: ATOMS' in line):
            if flag==0:
                for i in range(atom_number):
                        next_line=[s for s in(f.readline().split(' '))]
                        if int(next_line[1])==atom_ID:
                            counter=counter+1
                            starting_pos[int(next_line[0])-1]=[float(s) for s in next_line[2:5]]
                flag=1
                #starting_pos=starting_pos[np.all(starting_pos)];
                starting_pos=starting_pos[~np.all(np.isnan(starting_pos),axis=1)]
            else:
                for i in range(atom_number):
                    next_line=[s for s in(f.readline().split(' '))]
                    if int(next_line[1])==atom_ID:
                        temp[int(next_line[0])-1]=[float(s) for s in next_line[2:5]]
                deltax=0
                deltay=0
                deltaz=0
                temp=temp[~np.all(np.isnan(temp),axis=1)]
                for i in range(len(temp)):
                    deltax=deltax+(temp[i][0]-starting_pos[i][0])**2
                    deltay=deltay+(temp[i][1]-starting_pos[i][1])**2
                    deltaz=deltaz+(temp[i][2]-starting_pos[i][2])**2
                msds.append(deltax/atom_number+deltay/atom_number+deltaz/atom_number)
    return msds


def mean_squared_displacement_polymer(input_file):
    f=open(input_file)
    number_found=0
    atom_number=0
    step_count=0
    box_size_found=0
    mean_kuhn_lengths=0.0
    flag=0
    while (number_found==0 or box_size_found==0):
        if('ITEM: NUMBER OF ATOMS' in f.readline()):
            number_found=1
            atom_number=int(f.readline())
        if('ITEM: BOX BOUNDS' in f.readline()):
            box_size_found=1
            size=[float(s) for s in(f.readline().split(' '))]
            x_start=size[0]
            x_stop = size[1]
    kuhn_lengths=[]
    starting_pos=np.full([atom_number,3],np.nan)
    center_of_mass=np.zeros(3)
    center_of_masses=[]
    msds=[]
    for line in f: 
        temp=np.full([atom_number,3],np.nan)
        if('ITEM: ATOMS' in line):
            if flag==0:
                for i in range(atom_number):
                        next_line=[s for s in(f.readline().split(' '))]
                        starting_pos[int(next_line[0])-1]=[float(s) for s in next_line[2:5]]
                flag=1
                for i in range(atom_number):
                    center_of_mass=center_of_mass+starting_pos[i,:]
                center_of_mass=center_of_mass/atom_number
                for i in range(atom_number):
                    starting_pos[i,:]=starting_pos[i,:]-center_of_mass
            else:
                center_of_mass=np.zeros(3)
                for i in range(atom_number):
                    next_line=[s for s in(f.readline().split(' '))]
                    temp[int(next_line[0])-1,:]=[float(s) for s in next_line[2:5]]
                for i in range(atom_number):
                    center_of_mass=center_of_mass+temp[i,0:3]
                center_of_mass=center_of_mass/atom_number
                center_of_masses.append(center_of_mass)
                delta=np.zeros(3)
                for i in range(atom_number-1):
                    delta=delta+(temp[i,:]-center_of_mass-starting_pos[i,:])**2
                msds.append(np.sum(delta/atom_number))
    return msds,center_of_masses

def local_density(input_file,id1,id2,max_length=10000000):
    f=open(input_file)
    number_found=0
    atom_number=0
    temp={}
    borders=[]
    distances=[]
    step_count=0
    id1_com=np.zeros(3)
    box_size_found=0
    counter=0
    while (number_found==0 or box_size_found==0):
        if('ITEM: NUMBER OF ATOMS' in f.readline()):
            number_found=1
            atom_number=int(f.readline())
        if('ITEM: BOX BOUNDS' in f.readline()):
            box_size_found=1
            borders.append([float(s) for s in(f.readline().split(' '))])
            borders.append([float(s) for s in(f.readline().split(' '))])
            borders.append([float(s) for s in(f.readline().split(' '))])
    counter=0
    max_distance=np.sqrt(3)*np.abs((borders[0][0]-borders[0][1]))
    for line in f:
        temp[id1]=[]
        temp[id2]=[]
        if('ITEM: ATOMS' in line):
            counter=counter+1

            for i in range(atom_number):
                line=[float(s) for s in(f.readline().split(' '))]
                if(int(line[1])==id1 or int(line[1])==id2):
                    temp[int(line[1])].append(line[2:5])
            for i in range(len(temp[id1])):
                for j in range(len(temp[id2])):
                    distance12=distance_PBC(temp[id1][i],temp[id2][j],borders)
                    if distance12<max_distance:
                        distances.append(distance12)
                if(len(distances)>max_length):
                    return distances
    return distances

def second_virial_coefficient(alpha,N):
    return 5*((alpha**5-alpha**3-1/(5*alpha**3)))/(3*np.sqrt(N))

def density_fluctuation(input_file,box_size=30,particle_id=1,max_eval=1e6):
    f=open(input_file)
    number_found=0
    atom_number=0
    temp={}
    borders=[]
    distances=[]
    step_count=0
    box_size_found=0
    while (number_found==0 or box_size_found==0):
        if('ITEM: NUMBER OF ATOMS' in f.readline()):
            number_found=1
            atom_number=int(f.readline())
        if('ITEM: BOX BOUNDS' in f.readline()):
            box_size_found=1
            borders.append([float(s) for s in(f.readline().split(' '))])
            borders.append([float(s) for s in(f.readline().split(' '))])
            borders.append([float(s) for s in(f.readline().split(' '))])
    temp=[]
    max_lengths=[np.abs(i[1]-i[0]) for i in borders]
    sizes=[int(max_length)//int(box_size) for max_length in max_lengths]
    n_particles_temp=np.zeros(sizes)
    n_particles_temp_squared=np.zeros(sizes)
    n_particles=np.zeros(sizes)
    n_particles_squared=np.zeros(sizes)
    for line in f:
        temp=[]
        tempid=[]
        if('ITEM: ATOMS' in line):
            for i in range(atom_number):
                line=[float(s) for s in(f.readline().split(' '))]
                if line[int(1)]==particle_id:

                    temp=PBC_done_right([float(i)for i in line[2:5]],borders)
                    bin_count=[int(sizes[i]*int(temp[i]-borders[i][0])//int(max_lengths[i])) for i in range(len(temp))]
                    print([f'{temp[i]:.20f},{bin_count[i]}' for i in range(len(temp))])
                    n_particles_temp[bin_count[0]][bin_count[1]][bin_count[2]]=1+n_particles_temp[bin_count[0]][bin_count[1]][bin_count[2]]
        n_particles=n_particles+n_particles_temp
        n_particles_squared=n_particles_squared+n_particles_temp**2
        n_particles_temp=np.zeros(sizes)
    return n_particles,n_particles_squared

def plot_binary(ha1,ha2,y, title, draw_line=False, x_label='$\phi$',y_label='$\phi$'):
    collapsedIndexes=[i  for i in range(len(y)) if y[i]>1.8]
    collapsedIndexes=[i  for i in range(len(y)) if y[i]>1 and y[i]<1.8]
    plt.scatter([ha1[i] for i in collapsedIndexes],[ha2[i] for i in collapsedIndexes], marker='s',label='swollen',color='green')
    collapsedIndexes=[i  for i in range(len(y)) if y[i]<1]
    plt.scatter([ha1[i] for i in collapsedIndexes],[ha2[i] for i in collapsedIndexes], alpha=0.5,marker='o',label='collapsed',color='indigo')
    if draw_line:
        plt.plot([0,1],[1,0])
    plt.legend()
    plt.xlabel(f'{x_label} small crowders')
    plt.ylabel(f'{y_label} big crowders')
    #plt.title(title)

if __name__=="__main__":
    #figure1
    '''data=pd.read_csv('./data/dataFrameAlpha')
    df=data[(data['r']==0.5)&(data['R']==0.0)|(data['r']==0.5)&(data['R']==1.0)]
    data=data.drop('inputFile',axis=1)
    
    df=data.groupby(by=['n','N']).mean()
    params = {'lines.markersize':'10','font.size':'36','legend.fontsize':'x-large','figure.figsize':(10,8),'axes.labelsize':'20','axes.titlesize':'18','xtick.labelsize':'18','ytick.labelsize':'18','legend.fontsize':'20'}
    plt.rcParams.update(params)
    
    #figure1
    plt.scatter(data['exp_virial'], data['theory_virial'])
    plt.ylabel('second virial coefficient (theory)')
    plt.xlabel('second virial coefficient (simulation)')
    plt.xlim([-0.5,1.5])
    plt.ylim([-0.5,1.5]) 
    plt.plot([-0.5,1.5],[-0.5,1.5])
    plt.tight_layout()
    plt.savefig('./figures/second_virial_exp_vs_theory.png')
    plt.show()



    #figure2
    sizesCouples=df.groupby(by=['r','R'],as_index=False).count()
    sizesCouples=sizesCouples[['r','R']]
    plt.rcParams['figure.figsize'] = [13, 8]
    x=np.linspace(0,1.25,100)
    df=data[(data['r']==0.5) & (data['R']==1.0)].groupby(by=['n','N']).mean()
    ha_small = list(df['ha_small'])
    ha_big = list(df['ha_big'])
    y   = list(df['alpha'])
    R,r=(1.0,0.5)
    plot_binary(ha_small,ha_big,y,f'R={R}  r={r}', draw_line=True, x_label='$x_{ha}$', y_label='$x_{ha}$')
    plt.tight_layout()
    plt.savefig('./figures/phase_diagram_ternary.png')
    plt.show()
    


    #figure3
    x=np.linspace(0,1.25,100)
    df=data[(data['r']==0.5) & (data['R']==0.0)].groupby(by=['n','N']).mean()
    ha_small= list(df['ha_small'])
    ha_big  = list(df['ha_big'])
    x1      =[ha_small[i]+ha_big[i] for i in range(len(ha_small))]
    y1      = list(df['alpha'])
    df=data[(data['r']==0.5) & (data['R']==1.0)].groupby(by=['n','N']).mean()
    ha_small= list(df['ha_small'])
    ha_big  = list(df['ha_big'])
    x2      =[ha_small[i]+ha_big[i] for i in range(len(ha_small))]
    y2      = list(df['alpha'])
    plt.scatter(x1,y1, label='monodisperse crowders')
    plt.scatter(x2,y2,label='bidisperse crowders')
    plt.xlabel("$x_{ha}$")
    plt.ylabel("$\\alpha$")
    plt.title("")
    plt.legend()
    plt.tight_layout()
    plt.savefig('./figures/alpha_ternary_binary.png')
    print(df)
    plt.show()   


    #figure4
    t=50000000
    R0=np.sqrt(40*1.5**2)
    ys1 =np.array(gyration_radius("./data/dumpOutputR10r05/polydumpN110000M8500t002R10r05N1.lammpstrj",t,0))/R0
    ys2 =np.array(gyration_radius("./data/dumpOutputR10r05/polydumpN110000M8500t002R10r05N2.lammpstrj",t,0))/R0
    ys3 =np.array(gyration_radius("./data/dumpOutputR10r05/polydumpN110000M8500t002R10r05N3.lammpstrj",t,0))/R0
    max_range=min([len(ys1),len(ys2),len(ys3)])
    for i in range(0,max_range):
        ys1[i]=(ys1[i]+(i*ys1[i-1]))/(i+1)
        ys2[i]=(ys2[i]+(i*ys2[i-1]))/(i+1)
        ys3[i]=(ys3[i]+(i*ys3[i-1]))/(i+1)
    plt.plot(ys1[0:max_range:100],marker='.',label="simulation 1")
    plt.plot(ys2[0:max_range:100],marker='.',label="simulation 2")
    plt.plot(ys3[0:max_range:100],marker='.',label="simulation 3")
    plt.xlabel("$t$")
    plt.ylabel("$\\alpha$")
    plt.legend()
    plt.tight_layout()
    plt.savefig('./figures/equilibration.png')
    plt.show()


    #figure5
    V=60*60*60
    n_1     =40
    n_2     =3500
    n_3     =140000
    n_tot=n_1+n_2+n_3
    r_1     =1.5
    r_2     =1.0
    r_3     =0.5
    x_1     =n_1/n_tot
    x_2     =n_2/n_tot
    x_3     =n_3/n_tot
    phi_tot =(4*np.pi/(3*V))*(n_1*r_1**3+n_2*r_2**3+n_3*r_3**3)
    exp_x,exp_g=radial_distribution_done_right(f'./data/dumpOutputR10r05/alldumpN140000M3500t002R10r05N1.lammpstrj',2,3,n_bins=500,max_length=200000000)
    theory_x,theory_g=theoretical_functions.rdf_rfa(phi_tot,[x_1,x_2,x_3],[1,2/3,1/3],1,2)
    plt.plot(exp_x,exp_g)
    plt.plot(3*theory_x,theory_g)
    plt.xlabel('$r$')
    plt.ylabel('$g(r)$')
    plt.xlim([0,10])
    plt.tight_layout()
    plt.savefig('./figures/rdf_ternary.png')
    plt.show()

    #figure6
    V           =60*60*60
    n           =7000
    r           =1.5
    x           =1
    phi_tot     =(4*np.pi/(3*V))*(n*r**3)
    exp_x,exp_g =radial_distribution_done_right(f'./data/DumpOutputMonodisperse/alldumpN7000M0t002R15r05N1.lammpstrj',1,1,n_bins=500,max_length=200000000)


    theory_x,theory_g=theoretical_functions.rdf_rfa(phi_tot,[x,0,0],[1,1,1],1,2)
    theory_x_py=theory_x
    theory_g_py=[theoretical_functions.monodisperse_correlation_function(x,1,phi_tot) for x in theory_x_py]
    plt.plot(exp_x,exp_g, label='simulation')
    plt.plot(3.0*theory_x,theory_g,label='PY')
    plt.plot(3.0*theory_x_py,theory_g_py,label='RFA')
    plt.xlabel('$r$')
    plt.ylabel('$g(r)$')
    plt.xlim([0,10])
    plt.tight_layout()
    plt.legend()
    plt.savefig('./figures/rdf_monodisperse.png')
    plt.show()
    
    #figure7
    V           =60*60*60
    n           =140000
    r           =0.5
    x           =1
    phi_tot     =(4*np.pi/(3*V))*(n*r**3)
    exp_x,exp_g =radial_distribution_done_right(f'./data/dumpOutputR00r05/spheresdumpN140000M0t002R05r00N2.lammpstrj',2,2,n_bins=500,max_length=200000000)

    print(phi_tot)
    theory_x,theory_g=theoretical_functions.rdf_rfa(phi_tot,[x,0,0],[1,1,1],1,2)
    theory_x_py=theory_x
    theory_g_py=[theoretical_functions.monodisperse_correlation_function(x,1,phi_tot) for x in theory_x_py]
    plt.plot(exp_x,exp_g, label='simulation')
    #plt.plot(3.0*theory_x,theory_g,label='PY')
    plt.plot(theory_x_py,theory_g_py,label='RFA')
    plt.xlabel('$r$')
    plt.ylabel('$g(r)$')
    plt.xlim([0,5])
    plt.tight_layout()
    plt.legend()
    plt.savefig('./figures/rdf_monodisperse_poly.png')
    plt.show()
    '''

    a,b=density_fluctuation('./data/dumpOutputR00r05/polydumpN120000M0t002R00r05N2.lammpstrj',30,1,10000)
    print(a)
    print(b)
