import numpy as np
import palmerpenguins
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load Cost-Effectiveness Results
Results = pd.read_csv("/content/gdrive/MyDrive/Cost-Effectiveness_Report.csv")

# LIAR Dataset
F1_Score = Results['Liar - f1']
Training_time = Results['Liar - Time']
Models = Results['Models']
COLORS = ["#1B9E77", "#D95F02", "#7570B3", "#FFEFDB", "#00FFFF", "#458B74", "#E3CF57", "#000000", "#0000FF", "#8A2BE2", "#8B2323", "#98F5FF", "#7FFF00", "#EE7621", "#CAFF70", "#2F4F4F", "#FF1493", "#00688B", "#191970", "#FFFF00"]

F1_Score = pd.Series(F1_Score)
Training_time = pd.Series(Training_time)
Models = pd.Series(Models)
  
data = pd.DataFrame({'F1_Score': F1_Score, 'Training_time': Training_time, 'Models': Models})

sns.lmplot(x='Training_time', y='F1_Score', data=data, hue='Models', fit_reg=False)
plt.ylim([0, 1])
plt.axvline(x=Training_time.mean(), linewidth=1, color='k', linestyle='dotted')
plt.axhline(y=F1_Score.mean(), linewidth=1, color='k', linestyle='dotted')
plt.savefig('plot1.pdf', dpi=1000)
plt.show()

# ISOT Dataset
F1_Score = Results['ISOT - f1']
Training_time = Results['ISOT - Time']
Models = Results['Models']
COLORS = ["#1B9E77", "#D95F02", "#7570B3", "#FFEFDB", "#00FFFF", "#458B74", "#E3CF57", "#000000", "#0000FF", "#8A2BE2", "#8B2323", "#98F5FF", "#7FFF00", "#EE7621", "#CAFF70", "#2F4F4F", "#FF1493", "#00688B", "#191970", "#FFFF00"]

F1_Score = pd.Series(F1_Score)
Training_time = pd.Series(Training_time)
Models = pd.Series(Models)
  
data = pd.DataFrame({'F1_Score': F1_Score, 'Training_time': Training_time, 'Models': Models})

sns.lmplot(x='Training_time', y='F1_Score', data=data, hue='Models', fit_reg=False)
plt.ylim([0, 1])
plt.axvline(x=Training_time.mean(), linewidth=1, color='k', linestyle='dotted')
plt.axhline(y=F1_Score.mean(), linewidth=1, color='k', linestyle='dotted')
plt.savefig('plot2.pdf', dpi=1000)
plt.show()

# COVID Dataset
F1_Score = Results['Covid-2 - f1']
Training_time = Results['Covid-2 - Time']
Models = Results['Models']
COLORS = ["#1B9E77", "#D95F02", "#7570B3", "#FFEFDB", "#00FFFF", "#458B74", "#E3CF57", "#000000", "#0000FF", "#8A2BE2", "#8B2323", "#98F5FF", "#7FFF00", "#EE7621", "#CAFF70", "#2F4F4F", "#FF1493", "#00688B", "#191970", "#FFFF00"]

F1_Score = pd.Series(F1_Score)
Training_time = pd.Series(Training_time)
Models = pd.Series(Models)
  
data = pd.DataFrame({'F1_Score': F1_Score, 'Training_time': Training_time, 'Models': Models})

sns.lmplot(x='Training_time', y='F1_Score', data=data, hue='Models', fit_reg=False)
plt.ylim([0, 1])
plt.axvline(x=Training_time.mean(), linewidth=1, color='k', linestyle='dotted')
plt.axhline(y=F1_Score.mean(), linewidth=1, color='k', linestyle='dotted')
plt.savefig('plot3.pdf', dpi=1000)
plt.show()

# GM Dataset
F1_Score = Results['GM - f1']
Training_time = Results['GM - Time']
Models = Results['Models']
COLORS = ["#1B9E77", "#D95F02", "#7570B3", "#FFEFDB", "#00FFFF", "#458B74", "#E3CF57", "#000000", "#0000FF", "#8A2BE2", "#8B2323", "#98F5FF", "#7FFF00", "#EE7621", "#CAFF70", "#2F4F4F", "#FF1493", "#00688B", "#191970", "#FFFF00"]

F1_Score = pd.Series(F1_Score)
Training_time = pd.Series(Training_time)
Models = pd.Series(Models)
  
data = pd.DataFrame({'F1_Score': F1_Score, 'Training_time': Training_time, 'Models': Models})

sns.lmplot(x='Training_time', y='F1_Score', data=data, hue='Models', fit_reg=False)
plt.ylim([0, 1])
plt.axvline(x=Training_time.mean(), linewidth=1, color='k', linestyle='dotted')
plt.axhline(y=F1_Score.mean(), linewidth=1, color='k', linestyle='dotted')
plt.savefig('plot4.pdf', dpi=1000)
plt.show()



