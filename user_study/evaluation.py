import pandas as pd
import matplotlib.pyplot as plt

results = pd.read_csv('results.csv')  # 26 participants

""" Task 1"""

answers = [1, 1, 2, 1, 1, 2, 2, 2, 1, 1, 2, 1, 1, 1, 2]

idx = 0
score = 0
for key in results.keys()[2:17]:
    for item in results[key]:
        if item == 'Option ' + str(answers[idx]):
           score += 1
    idx += 1

ans = score / (len(answers) * len(results))
print("Result Task 1", ans)


""" Task 2"""

answers = [2, 1, 1, 2, 2, 1, 2, 1, 1, 2]

idx = 0
score = 0
for key in results.keys()[17:27]:
    for item in results[key]:
        if item == 'Option ' + str(answers[idx]):
           score += 1
    idx += 1

ans = score / (len(answers) * len(results))

print("Result Task 2", ans)


"""Task 3"""

x_axis = ['WAE/GAN', 'D-VAE/GAN', 'D-VAE']
y_axis = [141, 87, 30]  # the results are calculated before
bars = plt.bar(x_axis, y_axis, width=0.5)
plt.axhline(y=0.5, xmin=0, xmax=0.33, linewidth=1, color='k')
plt.axhline(y=0.2, xmin=0.33, xmax=0.66, linewidth=1, color='k')
plt.axhline(y=0.1, xmin=0.66, xmax=1.0, linewidth=1, color='k')
for i, bar in enumerate(bars):
    yval = bar.get_height()
    plt.text(bar.get_x() + 0.15, yval + 1, f'{y_axis[i]}')
plt.ylabel('Scores')
plt.title('Subjective assessment')
plt.show()


""" Task 4"""

answers = [1, 1, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 1, 2, 1]

idx = 0
score = 0
for key in results.keys()[57:72]:
    for item in results[key]:
        if item == 'Option ' + str(answers[idx]):
           score += 1
    idx += 1

ans = score / (len(answers) * len(results))

print("Result Task 4", ans)


""" Task 5"""

answers = [2, 1, 2, 2, 1, 2, 1, 1, 2, 1]

idx = 0
score = 0
for key in results.keys()[72:82]:
    for item in results[key]:
        if item == 'Option ' + str(answers[idx]):
           score += 1
    idx += 1

ans = score / (len(answers) * len(results))

print("Result Task 5", ans)



