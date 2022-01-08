import time

def recalculate_traj_energies(traj,calc=None,config=None,writer=None,rewrite_pbc=False):
	from copy import deepcopy
	import gpaw
	start_time = time.time()
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
			atoms.info['calculation_time'] = time.time()-start_time
			if writer is not None:
				writer.write(atoms)
			print(atoms)
			print(atoms.info)
		except gpaw.grid_descriptor.GridBoundsError:
			print(f'GridBounds box error for {i}th active learning')
		except gpaw.KohnShamConvergenceError:
			print(f'Convergence Error for {i}th active learning')
		# except:
		# 	print(f'Convergence error for {i}th active learning')
		

	# return traj_new