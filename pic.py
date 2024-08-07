import pandas as pd
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


clean = [0.926, 0.9220, 0.886, 0.837, 0.763, 0.665, 0.591, 0.508, 0.426, 0.371, 0.188, 0.123]
adv = [0.996, 0.341, 0.167, 0.0915, 0.0519, 0.0284, 0.0123, 0.0111, 0.008, 0.0037, 0.001237, 0.0]
idx = [1,2,3,4,5,6,7,8,9,10,15,20]

clean = [i*100 for i in clean]
adv = [(1-i)*100 for i in adv]



# pic
# vars_criteria = {'adversarial data': adv,
#                  'normal data': clean}

df=pd.DataFrame(dict(x=idx, y=clean))

sns.set_theme(style='darkgrid')  # 图形主题
plt.figure(1)
sns.lineplot(data=df, linewidth=2.5, x="x", y="y")
df=pd.DataFrame(dict(x=idx, y=adv))
sns.lineplot(data=df, linewidth=2.5, x="x", y="y")


plt.xlabel("Adversarial steps")
plt.ylabel("Accuracy")

plt.legend(labels=['normal data', 'adversarial data'])
plt.savefig('sst2_boundary.pdf')
plt.show()