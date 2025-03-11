import os

directory = '/work/mech-ai/jrrade/Protein/AF_swissprot_py3d_virtual_afm/'
# dirs = os.listdir(directory)

# train_dirs = dirs[:int(0.75*len(dirs))]
# val_dirs = dirs[int(0.75*len(dirs)):int(0.9*len(dirs))]
# test_dirs = dirs[int(0.9*len(dirs)):]

# with open('/work/mech-ai/jrrade/Protein/upfusion/data/data_split/train_samples.txt', 'w') as f:
# 	for d in train_dirs:
# 		f.write(os.path.join(directory,d))
# 		f.write('\n')

# with open('/work/mech-ai/jrrade/Protein/upfusion/data/data_split/val_samples.txt', 'w') as f:
# 	for d in val_dirs:
# 		f.write(os.path.join(directory,d))
# 		f.write('\n')


# with open('/work/mech-ai/jrrade/Protein/upfusion/data/data_split/test_samples.txt', 'w') as f:
# 	for d in test_dirs:
# 		f.write(os.path.join(directory,d))
# 		f.write('\n')


with open('/work/mech-ai/jrrade/Protein/ProFusion/data/data_split/train_samples.txt', 'r') as f:
	train_dirs = f.readlines()
	train_dirs = [d.strip() for d in train_dirs]

with open('/work/mech-ai/jrrade/Protein/ProFusion/data/data_split/val_samples.txt', 'r') as f:
	val_dirs = f.readlines()
	val_dirs = [d.strip() for d in val_dirs]

with open('/work/mech-ai/jrrade/Protein/ProFusion/data/data_split/test_samples.txt', 'r') as f:
	test_dirs = f.readlines()
	test_dirs = [d.strip() for d in test_dirs]

#generate metadata with different number of samples ['256', '1k', '10k', '50k', '100k', '150k']
num_samples = {'256':256, '1k':1000, '10k':10000, '50k':50000, '100k':100000, '150k':150000}

for num in num_samples:
	with open('/work/mech-ai/jrrade/Protein/upfusion/data/data_split/train_samples_'+num+'.txt', 'w') as f:
		for d in train_dirs[:num_samples[num]]:
			f.write(os.path.join(directory,d))
			f.write('\n')
	
	with open('/work/mech-ai/jrrade/Protein/upfusion/data/data_split/val_samples_'+num+'.txt', 'w') as f:
		for d in val_dirs[:num_samples[num]]:
			f.write(os.path.join(directory,d))
			f.write('\n')
	
	with open('/work/mech-ai/jrrade/Protein/upfusion/data/data_split/test_samples_'+num+'.txt', 'w') as f:
		for d in test_dirs[:num_samples[num]]:
			f.write(os.path.join(directory,d))
			f.write('\n')