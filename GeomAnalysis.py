#!/usr/bin/env python

import numpy as np, os
import numpy.linalg as npl

molGeom = open('.\\Input_Files\\geom.txt', 'r')
geomCont = molGeom.readlines()


atomCount = int(geomCont[0].replace('\n', ''))
geomCont = geomCont[1:]

def pos(positions):

	posMatrix = np.array([])

	for i in positions:

		i = i.split('\t')
		i[-1] = i[-1].replace('\n','')
		i = i[1:]
		i = [float(x) for x in i]
		posMatrix = np.append(posMatrix, i)

	posMatrix = np.reshape(posMatrix, (len(positions),3))
	return posMatrix


def bondLenghts(positions):

	distances = np.array([])
	posDummy = positions.copy()
	bondPartnerCount = 0			#correctly counts the index of the bonding partner as posDummy gets shorter while looping
	
	for i in range(len(positions)):

		posDummy = posDummy[1:]

		bondPartnerCount = i+1

		for k in range(len(posDummy)):

			distances = np.append(distances, i)
			distances = np.append(distances, bondPartnerCount)
			distances = np.append(distances, np.abs(npl.norm(positions[i]-posDummy[k])))
			bondPartnerCount += 1
	
	distances = np.reshape(distances, (-1,3))
	return distances



def bondAngles(positions, distances):

	"""
	Now gives the correct angles. However it needs some optimization as there are 5 for loops which can probably cut down to 3
	"""

	bondDis = [distances[x] for x in range(len(distances)) if distances[x][2] < 4]
	bondDis = np.reshape(bondDis, (-1,3))
	angles = np.array([])

	for count in range(len(bondDis)):
		for j in range(1,atomCount-1):
			for i in range(0,j):

				if(j == i):
					continue

				elif((i == bondDis[count][0] or i == bondDis[count][1]) and (j == bondDis[count][0] or j == bondDis[count][1])):
					D_ij = bondDis[count][2]
					e_ij = (positions[j]-positions[i])/D_ij

					for k in range(j,atomCount):


						if(k == j or k == i):
							continue

						for count2 in range(len(bondDis)):
							if((j == bondDis[count2][0] or j == bondDis[count2][1]) and (k == bondDis[count2][0] or k == bondDis[count2][1])):
								D_jk = bondDis[count2][2]
								e_jk = (positions[k]-positions[j])/D_jk

								angle = 180 - np.arccos(np.dot(e_ij,e_jk))*180/np.pi
								angles = np.append(angles, [i,j,k,angle])
								angles = np.reshape(angles, (-1,4))
								
	return angles


def main(positions):

	posMain = pos(positions)
	lenghts = bondLenghts(posMain)
	angles = bondAngles(posMain,lenghts)

	print("Number of atoms: ")
	print(atomCount)
	print("\nPositions:")
	print(posMain)
	print("\nBond lenghts:")
	print(lenghts)
	print("\nBond Angles:")
	print(angles)

	return None

#bondAngles(pos(geomCont),bondLenghts(pos(geomCont)))

main(geomCont)
molGeom.close()