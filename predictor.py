from scipy import misc
import os
import imageutil
import matplotlib.pyplot as plt
import cnnmodel
import numpy as np

sample_path = 'th_samples'

def evaluate(model):

	paths = os.listdir(sample_path)
	preview = False

	# -----------------------------
	# Init Statistic Variables
	# -----------------------------

	# { character : [count, right] }
	classes = [chr(i) for i in range(ord('ก'), ord('ฮ')+1)]
	classes_dict = {chr(i):[0,0] for i in range(ord('ก'), ord('ฮ')+1)}

	correct_count = 0
	test_data_count = 0
	subplot_num = 0

	# -----------------------------

	for character in paths:
		# ignore system files
		if(character.startswith('.')):
			continue	

		img_paths = os.listdir(sample_path+"/"+character)

		for img_name in img_paths:
			# ignore system files
			if(img_name.startswith('.')):
				continue	

			test_data_count += 1

			if preview:
				subplot_num += 1
				if subplot_num <= 9:
					plt.subplot(3, 3, subplot_num)
				else:
					subplot_num = 0
					plt.show()

			img = imageutil.readimageinput(sample_path+"/"+character+'/'+img_name, preview=preview, invert=False, size=(128,128))

			pred = model.predict_classes(img)

			pred_proba = model.predict_proba(img)
			pred_proba = "%.2f%%" % (pred_proba[0][pred]*100)

			pred_class = classes[pred[0]]

			is_correct = str(pred_class) == str(character)

			classes_dict[character][0] += 1
			if is_correct:
				correct_count += 1
				classes_dict[character][1] += 1

			result_sum = "ans: {} predicted: {} with probability {} | {}".format(character, str(pred_class), pred_proba, "correct" if is_correct else "INCORRECT")

			print(result_sum)

			if preview:
				plt.title("pred: {}".format(pred_class), fontproperties='Tahoma', color='black' if is_correct else 'red')

	classes_acc = {k:(classes_dict[k][1]/classes_dict[k][0] if classes_dict[k][0] > 0 else 0) for k in classes_dict}
	print(classes_acc)
	print('{}/{} correct ({})'.format(correct_count, test_data_count, correct_count/test_data_count))

	fig = plt.figure()
	ax = fig.gca()
	d = classes_acc
	X = np.arange(len(d))
	C = [
		'g' if classes_acc[k] >= 0.7 else 
		('y' if classes_acc[k] >= 0.5 else 'r') 
		for k in classes_acc
	]
	plt.bar(X, d.values(), color=C, align='center', width=0.5)
	plt.axhline(0.7, color='g', linestyle='dashed', linewidth=1)
	plt.axhline(0.5, color='y', linestyle='dashed', linewidth=1)
	plt.xticks(X, d.keys(), fontname='Tahoma')
	plt.ylim(0, 1.1)
	plt.show()

if __name__ == "__main__":
	model_name = 'model_all_char'
	model = cnnmodel.load_model_from_json(model_name+'.json', model_name+'.h5')
	print('model loaded from disk')
	evaluate(model)
