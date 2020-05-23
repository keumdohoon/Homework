#3장 데이터 시각화
#p.49_visualizing data
from matplotlib import pyplot as plt

years = [1950, 1960, 1970, 1980, 1990, 2000, 2010]
gdp = [300.2, 543.3, 1075.9, 2862.5, 5979.6, 10289.7, 14958.3]

# x축=연도, y축에는 gdp가있는 선 그래프를 만든다. 
plt.plot(years, gdp, color='green', marker='o', linestyle='solid')
#제목을 더한다. 
plt.title("Nominal GDP")

#  y-axis 에 레이블을 만들어준다. 
plt.ylabel("Billions of $")
plt.show()


plt.savefig('im/viz_gdp.png')
plt.gca().clear()

