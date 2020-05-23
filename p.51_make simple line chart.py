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

#p.51_make simple line chart

movies = ["Annie Hall", "Ben-Hur", "Casablanca", "Gandhi", "West Side Story"]
num_oscars = [5, 11, 3, 8, 10]

# 막대의x좌표는 [0, 1, 2, 3, 4], y좌표는 [num_oscars] 로 설정
plt.bar(range(len(movies)), num_oscars)

plt.title("My Favorite Movies")     # 제목추가
plt.ylabel("# of Academy Awards")   # y축에 레이블 추가

# x축에 각 막대의 중앙에 영화제목을 레이블로 추가. 
plt.xticks(range(len(movies)), movies)

plt.show()

from collections import Counter
grades = [83, 95, 91, 87, 70, 0, 85, 82, 100, 67, 73, 77, 0]

# 점수를 10점 단위로 그룹화하고 100점은 90점대에 속하게 해준다. 
histogram = Counter(min(grade // 10 * 10, 90) for grade in grades)

plt.bar([x + 5 for x in histogram.keys()],  # 각막대를 오른쪽으로 5만큼 옮긴다. 
        histogram.values(),                 # 각 막대의 높이를 정해주고.
        10,                                 # 너비는 10으로 설정
        edgecolor=(0, 0, 0))                # 각 막대의 테두리는 검정으로 설정

plt.axis([-5, 105, 0, 5])                  # x축은-5 부터 105,
                                           # y축은 0 부터 5

plt.xticks([10 * i for i in range(11)])    # x축의 레이블은 0, 10, ..., 100
plt.xlabel("Decile")
plt.ylabel("# of Students")
plt.title("Distribution of Exam 1 Grades")
plt.show()



mentions = [500, 505]
years = [2017, 2018]

plt.bar(years, mentions, 0.8)
plt.xticks(years)
plt.ylabel("누군가가 '데이터 사이언스'라고 말한 #횟수")

# 이렇게 하지 않으면 matplotlib가 x축에 0,1레이블을 달고. 
#주변부 어딘가에는 +2.013e3 이라고 표기할 것이기에 별로 좋지 않은 matplotlib가 될것이다. 
plt.ticklabel_format(useOffset=False)

# 오해를 불러 일으키는  y축은500 이상의 부분만 보여줄것이다. 
plt.axis([2016.5, 2018.5, 499, 506])
plt.title("Look at the 'Huge' Increase!")
plt.show()

plt.axis([2016.5, 2018.5, 0, 550])
plt.title("Not So Huge Anymore")
# plt.show()
