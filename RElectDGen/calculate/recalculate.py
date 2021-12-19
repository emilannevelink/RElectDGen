

def recalculate_traj_energies(traj,calc=None,config=None,writer=None,rewrite_pbc=False):
	from copy import deepcopy
	import gpaw
	# traj_new = []
	if calc==None:
		from .calculator import oracle_from_config
	else:
		calci = calc
	for i, atoms in enumerate(traj):
		if rewrite_pbc:
			atoms.set_pbc(True)

		if calc==None:
			calci = oracle_from_config(config,atoms=atoms)
		atoms.calc = calci
		try:
			atoms.get_forces()
			if writer is not None:
				writer.write(atoms)
		except gpaw.grid_descriptor.GridBoundsError:
			print(f'GridBounds box error for {i}th active learning')
		except gpaw.KohnShamConvergenceError:
			print(f'Convergence Error for {i}th active learning')
		# except:
		# 	print(f'Convergence error for {i}th active learning')
		

	# return traj_new