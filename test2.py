#working

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import streamlit as st
import streamlit.components.v1 as comp



global pvectors
pvectors=[]


dim = int(st.number_input("What Dimensional vectors you wanna work with?  \n e.g. Enter 2 for 2D vector space "))
#c1,c2 = st.columns([0.5,0.5])
c3 = st.sidebar

slider = c3.slider("Zoom in/out", min_value=0, max_value=100, step=2, value=10)


def  plot_vector(origin,vector):
	pvectors.append([origin,vector])
	if dim!=2:
		return "Vector is not in 2D to plot."
	else:
		return ax1.quiver(origin[0], origin[1], vector[0], vector[1], angles='xy', scale_units='xy', scale=1, color=np.random.rand(10, 3), label='Vector 1')



def animate_vector(start,end):
	line = ax2.quiver(start[0], start[1], end[0], end[1], angles='xy', scale_units='xy', scale=1, color=np.random.rand(10, 3), label='Vector 1')
	def update(frame):
		x = np.linspace(start[0],end[0],100)
		y = np.linspace(start[1],end[1],100)
		line.set_UVC(x[frame],y[frame])
		return
	ani = FuncAnimation(fig, update, frames=100, interval=5, blit=False)
	comp.html(ani.to_jshtml(),height=900)


def gen_line(comb_vector):
	return ax2.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1, color=np.random.rand(10, 3), label='Vector 1')
   

def animate_all(pvectors):
	frame_list = [i*100 for i in range(len(pvectors))]
	lines = dict(map(lambda idx : ('line{}'.format(idx[0]), gen_line(idx[1])), enumerate(pvectors)))
	def update(frame):
		for num in frame_list:
			if (frame>=num) and (frame<num+100):
				vec_num = int(num/100)
				frame = frame - vec_num*100
				vector = pvectors[vec_num]
				x = np.linspace(vector[0][0],vector[1][0],100)
				y = np.linspace(vector[0][1],vector[1][1],100)
				lines[f'line{vec_num}'].set_UVC(x[frame],y[frame])
				return
	ani = FuncAnimation(fig,update,frames=100*len(pvectors),interval=5,blit=False, repeat=False)
	return ani


def transform(vector,tmatrix):
	vector = vector*tmatrix
	return np.array(vector)


def inner_product(v1,v2):
	return sum(v1*v2)  
 
def norm(vector):
	return np.sqrt(inner_product(vector,vector))

def distance(v1,v2):
	if len(v1) == len(v2):
		return norm(v1-v2)
	else:
		return "Both vectors must have same dimensions."

def angle(v1,v2):
	if len(v1) == len(v2):
		x = inner_product(v1,v2)/(norm(v1)*norm(v2))
		return np.arccos(x)*180/np.pi
	else:
		return "Both vectors must have same dimensions. "

def is_independent(vectors):
	matrix = []
	for vec in vectors.values():
		if len(vectors) > dim:
			return f"maximum of vectors needed to be linearly independent is {dim}, but {len(vectors)} were given."
		matrix.append(vec)
	st.write(matrix)
	if np.linalg.det(matrix)==0:
		return False
	else:
		 return True


def  orthonormal(vectors):
	if len(vectors)!=dim:
		st.write(f"Number of vectors needed for orthonormal vector transformation is {dim}, but {len(vectors)} were given.")
		return None
	elif is_independent(vectors)==True:
		orthonormal_set = {}
		q1 = vectors['vector_1']
		q11 = q1/np.sqrt(inner_product(q1,q1))
		orthonormal_set['orth_vector_1'] = q11
		for num in range(2,len(vectors)+1):
			new_vec = vectors[f'vector_{num}']
			for old_vec in orthonormal_set.values():
				new_vec = new_vec - (inner_product(new_vec,old_vec)*old_vec)
			new_vec1 = new_vec / np.sqrt(inner_product(new_vec, new_vec))
			orthonormal_set[f'orth_vector_{num}'] = new_vec1
		return orthonormal_set
	else:
		return None



# Create a figure and axis
fig, ax1 = plt.subplots()
fig, ax2 = plt.subplots()


# Set the limits of the plot
ax1.set_xlim([-slider, slider])
ax1.set_ylim([-slider, slider])
ax2.set_xlim([-slider, slider])
ax2.set_ylim([-slider, slider])
# Add grid and labels

#set labels and lining
ax1.grid(True)
ax1.axhline(0, color='black', linewidth=0.5)
ax1.axvline(0, color='black', linewidth=0.5)
ax1.set_xlabel('X-axis')
ax1.set_ylabel('Y-axis')   
ax2.grid(True)
ax2.axhline(0, color='black', linewidth=0.5)
ax2.axvline(0, color='black', linewidth=0.5)
ax2.set_xlabel('X-axis')
ax2.set_ylabel('Y-axis')







#test

origin = np.array([0,0])
vector = np.array([5,0])
matrix = np.matrix([[-2,1],
					[3,2]])
trans_vec = transform(vector,matrix)




vec_list = {}
num = int(c3.number_input("Enter number of input vectors for plotting."))


for i in range(num):
	ivector = c3.text_input(f'vector_{i+1}')
	if ivector=='':
			raise ValueError(f"Enter vector {num}")
	vec_list[f'vector_{i+1}'] = np.array(ivector.split(','),dtype=float)
	if len(vec_list[f'vector_{i+1}'])!=dim:
			raise ValueError(f"Vectors must be from R{dim} space.")
	if dim==2:
			plot = c3.checkbox('Plot & Animate',key=f'ani_{i}',value = False)
			if plot:
				plot_vector(origin,np.array(ivector.split(','),dtype=float))
			animate = c3.toggle('animate it?',key=i)
			if animate:
				animate_vector(origin,np.array(ivector.split(','),dtype=float))
		
	
with st.expander("Inner Products"):
	ip1 = st.radio('choose 1',options = vec_list.keys(),key=f'first')
	ip2 = st.radio('choose 1',options = vec_list.keys(),key=f'second')
	if st.button('Calculate'):
		ip = inner_product(vec_list[ip1],vec_list[ip2])
		st.write(f'Inner product of {ip1} and {ip2} is {ip}')


with st.expander("Norm"):
	ip = st.radio('Choose vector to find Norm', options=vec_list.keys(), key = 'norm')
	if ip:
		st.write(norm(vec_list[ip]))

with st.expander("Distance"):
	ip1 = st.radio('Choose vector to find Distance betweem them', options=vec_list.keys(), key = 'd1')
	ip2 = st.radio('Choose vector to find Distance betweem them', options=vec_list.keys(), key = 'd2')
	if ip1 and ip2:
		st.write(distance(vec_list[ip1],vec_list[ip2]))


with st.expander("Angle"):
	ip1 = st.radio('Choose vector to find angle betweem them', options=vec_list.keys(), key = 'a1')
	ip2 = st.radio('Choose vector to find angle betweem them', options=vec_list.keys(), key = 'a2')
	if ip1 and ip2:
		st.write(angle(vec_list[ip1],vec_list[ip2]))

with st.expander("Linearly Dependent/Independent"):
	li = is_independent(vec_list)
	if li==True:
		'Given vectors are linearly Independent'
	elif li==False:
		"Given vectors are linearly Dependent"
	else:
		li

with st.expander("Orthonormal Vectors"):
	orth = orthonormal(vec_list)
	if st.button("Show Orthogonal Vectors"):
		st.write(orth)

	if dim!=2:
		st.write("Entered vectors are not 2 dimensional to plot.")
	elif len(orth) != 0 and dim==2:
		for i in orth:
			st.write(i[5:])
			st.write(f'Transform it from {vec_list[i[5:]]} to {orth[i]}')
			if st.toggle('Transform and Animate',key = f'Trans{i}'):
				plot_vector(vec_list[i[5:]],orth[i])
	


if len(pvectors)!=0:
	dr = st.empty()
	dr.pyplot(ax1.figure)



ani = animate_all(pvectors)
if c3.button('Animate All'):
	comp.html(ani.to_jshtml(), height = 900)
#st.pyplot(ax2.figure)


