import MDAnalysis as mda 

universe = mda.Universe("step3_input.psf", "step3_input.dcd")

# Loading a structure or trajectory


u = mda.Universe("step3_input.psf", "step3_input.dcd")
print(u)
print(len(u.trajectory))

# Groups of atoms
print(u.residues)
u.atoms

# AtomGroup positions and methods
ca = u.select_atoms("protein")
print(ca.positions)
print(ca.positions.shape)
print(ca.center_of_mass())
print(ca.center_of_geometry())
print(ca.total_mass())
print(ca.total_charge())
print(ca.radius_of_gyration())
print(ca.bsphere())


# Working with trajectories
print(len(u.trajectory))

for ts in u.trajectory[:1]:
    time = u.trajectory.time
    rgyr = u.atoms.radius_of_gyration()
    print(f"Frame: {ts.frame:3d}, Time: {time:4.0f} ps, Rgyr: {rgyr:.4f} A")

print(u.trajectory.frame)
print(u.trajectory[1].frame)
frame = u.trajectory.frame
time = u.trajectory.time
rgyr = u.atoms.radius_of_gyration()
print("Frame: {:3d}, Time: {:4.0f} ps, Rgyr: {:.4f} A".format(frame, time, rgyr))

rgyr = []
time = []
protein = u.select_atoms("protein")
for ts in u.trajectory:
    time.append(u.trajectory.time)
    rgyr.append(protein.radius_of_gyration())

import pandas as pd
from matplotlib import pyplot as plt
rgyr_df = pd.DataFrame(rgyr, columns=['Radius of gyration (A)'], index=time)
rgyr_df.index.name = 'Time (ps)'

rgyr_df.head()

rgyr_df.plot(title='Radius of gyration')
plt.show()

# Dynamic selection

#dynamic = u.select_atoms('around 2 resname ALA', updating=True)
#print(type(dynamic))
#dynamic
#static = u.select_atoms('around 2 resname ALA')
#print(type(static))
#static

#u.trajectory.next()
#dynamic
#static

# Writing out coordinates

a = u.select_atoms('protein')
with mda.Writer('4pti.xtc', ca.n_atoms) as w:
    for ts in u.trajectory:
        w.write(ca)

from MDAnalysis.analysis import rms

bb = u.select_atoms('protein')

u.trajectory[0] # first frame
first = bb.positions
u.trajectory[-1] #last frame
last = bb.positions
rms.rmsd(first, last)
u.trajectory[0] # set to first frame

rmsd_analysis = rms.RMSD(u, select='backbone', groupselections=['protein'])
rmsd_analysis.run()
print(rmsd_analysis.results.rmsd.shape)


import pandas as pd

rmsd_df = pd.DataFrame(rmsd_analysis.results.rmsd[:, 2:],
                       columns=['Backbone', 'Protein'],
                       index=rmsd_analysis.results.rmsd[:, 1])
rmsd_df.index.name = 'Time (ps)'
rmsd_df.head()
rmsd_df.plot(title='RMSD')
plt.show()

import pickle
import numpy as np
np.set_printoptions(linewidth=100)

from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis

hbonds = HydrogenBondAnalysis(universe=u)

protein_hydrogens_sel = hbonds.guess_hydrogens("protein")
protein_acceptors_sel = hbonds.guess_acceptors("protein")

print(f"hydrogen_sel = {protein_hydrogens_sel}")
print(f"acceptors_sel = {protein_acceptors_sel}")

hbonds = HydrogenBondAnalysis(
    universe=u,
    donors_sel=None,
    hydrogens_sel=protein_hydrogens_sel,
    acceptors_sel=protein_acceptors_sel,
    d_a_cutoff=3.0,
    d_h_a_angle_cutoff=150,
    update_selections=False
)

hbonds.run(
    start=None,
    stop=None,
    step=None,
    verbose=True
)

print(hbonds.results.hbonds.shape)
print(hbonds.results.hbonds[0])
hbonds.results.hbonds.dtype
first_hbond = hbonds.results.hbonds[0]
frame, donor_ix, hydrogen_ix, acceptor_ix = first_hbond[:4].astype(int)
u.trajectory[frame]
atoms = u.atoms[[donor_ix, hydrogen_ix, acceptor_ix]]
atoms

# Counts the number of hydrogen bonds for each frame
plt.plot(hbonds.times, hbonds.count_by_time(), lw=2)

plt.title("Number of hydrogon bonds over time", weight="bold")
plt.xlabel("Time (ps)")
plt.ylabel(r"$N_{HB}$")

plt.show()

hbonds.count_by_type()

for donor, acceptor, count in hbonds.count_by_type():

    donor_resname, donor_type = donor.split(":")
    n_donors = u.select_atoms(f"resname {donor_resname} and type {donor_type}").n_atoms

    # average number of hbonds per donor molecule per frame
    mean_count = 2 * int(count) / (hbonds.n_frames * n_donors)  # multiply by two as each hydrogen bond involves two water molecules
    print(f"{donor} to {acceptor}: {mean_count:.2f}")

hbonds.count_by_ids()
counts = hbonds.count_by_ids()
most_common = counts[0]

print(f"Most common donor: {u.atoms[most_common[0]]}")
print(f"Most common hydrogen: {u.atoms[most_common[1]]}")
print(f"Most common acceptor: {u.atoms[most_common[2]]}")

np.save("4pti_hbonds.npy", hbonds.results.hbonds)

# Dihedral Angles

from MDAnalysis.analysis.dihedrals import Dihedral

tyr = u.select_atoms('resname TYR')

print(tyr)

results = []

# Open the result file to write
with open("results.txt", "w") as f:
    f.write("Timestep\tTyr1 Dihedral\tTyr2 Dihedral\tTyr3 Dihedral\tTyr4 Dihedral\n")  # Header

    # Iterate over each timestep
    for ts in u.trajectory:
        # Define indices for each tyrosine
        tyrosines = [
            [146, 148, 151, 152],
            [330, 332, 335, 336],
            [371, 373, 376, 377],
            [351, 353, 356, 357]
        ]

        # Initialize a list to store dihedral angles for each tyrosine
        dihedrals = []

        # Calculate dihedral angles for each tyrosine
        for indices in tyrosines:
            atom_selection = u.atoms[indices]
            dihedral = atom_selection.dihedral.value()
            dihedrals.append(dihedral)

        # Write timestep and dihedral angles to the result file
        f.write(f"{ts.frame}\t\t{dihedrals[0]}\t\t{dihedrals[1]}\t\t{dihedrals[2]}\t\t{dihedrals[3]}\n")

        # Calculate some result for the timestep (example)
        result_for_timestep = ts.frame * 2
        results.append(result_for_timestep)