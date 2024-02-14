import json
import matplotlib.pyplot as plt

results_dir = '/home/travail/Code/Some_Results/evolution_resol.json'

with open(results_dir, "r") as file:
    new_res = json.load(file)

name = 'classic_noDataAugmentation_noDepth_resol20'
resols = []
acc_means = []
acc_sub1, acc_sub2, acc_sub3, acc_sub4, acc_sub5 = [], [], [], [], []

for key, value in new_res.items():
    if key == name:
        for resol, subjects in value.items():
            resols.append(resol)
            acc_mean = 0
            for subject, acc in subjects.items():
                acc_mean += acc/5
            
            acc_sub1.append(subjects['1'])
            acc_sub2.append(subjects['2'])
            acc_sub3.append(subjects['3'])
            acc_sub4.append(subjects['4'])
            acc_sub5.append(subjects['5'])

            acc_means.append(acc_mean)

plt.plot(resols, acc_means)

plt.title('Evolution de la précision en fonction de la diminution de la résolution')
plt.xlabel('Résolution')
plt.ylabel('Précision moyennes sur les 5 sujets de tests')

plt.ylim(ymin=0.65, ymax=1)
plt.legend()

max_y = max(acc_means)
plt.annotate(f'Max: {round(0.93,2)}', xy=(0, max_y), xytext=(20, 0), textcoords='offset points')#replace 0.93 by max_y

plt.show()

plt.plot(resols, acc_sub1, label='Sujet 1')
plt.plot(resols, acc_sub2, label='Sujet 2')
plt.plot(resols, acc_sub3, label='Sujet 3')
plt.plot(resols, acc_sub4, label='Sujet 4')
plt.plot(resols, acc_sub5, label='Sujet 5')

plt.title('Evolution de la précision en fonction de la diminution de la résolution par sujets')
plt.xlabel('Résolution')
plt.ylabel('Précision')

plt.ylim(ymin=0.65, ymax=1)
plt.legend()

plt.show()