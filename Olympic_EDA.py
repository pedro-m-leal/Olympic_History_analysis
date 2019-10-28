"""

Import libraries and datasets

"""
#Libraries for Data Visualization

import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import numpy as np 
import seaborn as sns
import time
import pandas_profiling as pp

#Import the database

athlete_df=pd.read_csv('athlete_events.csv')
noc_df=pd.read_csv('noc_regions.csv')
continent_df=pd.read_csv('continents.csv')

"""

Processing the data

"""


#Modify Singapore's noc code - it should be SGP and not SIN

#noc_df[noc_df['NOC']=='SIN']=['SGP']
noc_df.replace('SIN','SGP',inplace=True)
continent_df.replace('SIN','SGP',inplace=True)
noc_df['region'].replace('USA', 'United States',inplace=True)


#Create an initial profile for the athletes dataframe

try:
    fn=open("df_pandas_profiling_OH.html")
    fn.close()
except:
    print("Creating a profile analysis of the dataset")
    profile = athlete_df.profile_report(title='Olympic History Profiling Report')
    profile.to_file(output_file="df_pandas_profiling_OH.html")

"""

Initial visuals together with describe() and info()


"""


print(athlete_df.describe())



# ## #Analyse the difference between Team and NOC

# noc_dict=noc_df[['NOC','region']].set_index('NOC',drop=True).to_dict()['region']
# print(athlete_df[athlete_df['NOC']=='SGP']['Team'].value_counts())
# print(athlete_df[athlete_df['Team']=='Singapore']['NOC'].value_counts())

# athlete_df['Region']=[noc_dict[i] for i in athlete_df['NOC']]

# mismatch_team_noc=athlete_df[athlete_df['Team']!=athlete_df['Region']]
# print(mismatch_team_noc[['Team','Region']])

# print(mismatch_team_noc['Region'].value_counts())
# print(mismatch_team_noc['Team'].value_counts())

### Separate the Games feature into Year_2 and Season_2 and check if these coincide with the already given labels for these features

athlete_df['Year_2']=[int(i.split(' ',1)[0]) for i in athlete_df['Games']]
athlete_df['Season_2']=[i.split(' ',1)[1] for i in athlete_df['Games']]


same_year=athlete_df['Year'].equals(athlete_df['Year_2'])
same_season=athlete_df['Season'].equals(athlete_df['Season_2'])
print('Are the Year and Year_2 collumn equal?\n')
print('Yes!\n') if same_year==True else print('No...\n')

print('Are the Season and Season_2 collumn equal?\n')
print('Yes!\n') if same_season==True else print('No...\n')

athlete_df.drop(['Games','Year_2','Season_2'],inplace=True,axis=1) if (same_year==True and same_season==True) else print('Nothing deleted')


### Replace non-medals with No_medal

athlete_df['Medal'].fillna(value='no_medal',inplace=True)


### Add 'region' feature

athlete_df=pd.merge(athlete_df,noc_df[['NOC','region']],on='NOC',how='left')




### Group countries by continent

athlete_df=pd.merge(athlete_df,continent_df[['NOC','Continent']],on='NOC',how='left')

print('\n\n---> Number of rows for which the continent is UNK: {}, which corresponds to {:.2f}% of the data. \n\n'.format(athlete_df[athlete_df['Continent']=='UNK'].shape[0],athlete_df[athlete_df['Continent']=='UNK'].shape[0]/athlete_df.shape[0]*100))

print(athlete_df[athlete_df['Continent'].isnull()][['NOC','Continent']])

print('\n\n=========== Unknown continent athletes ============ \n\n')
print(athlete_df[athlete_df['Continent']=='UNK'][['Name','Sport','Team','NOC','Continent','Medal','Year']])

print('\n\n===================== Dataset with Continent label =====================\n\n')

print(athlete_df.head(10))

"""

Alternative, less efficient version of adding continent.

noc_1=set(athlete_df['NOC'].tolist())
noc_2=set(continent_df['NOC'].tolist())

missing_noc=list(noc_1-noc_2)

print("\n\n---> These codes are missing, assign manually: {}.\n\n".format(missing_noc))


country_to_continent=continent_df[['NOC','Continent']].set_index('NOC',drop=True).to_dict()['Continent']


athlete_df['Continent2']=['UNK' if i in missing_noc else country_to_continent[i] for i in athlete_df['NOC'] ]


print('!!!!=====\n:',athlete_df[athlete_df['Continent2']!=athlete_df['Continent']])"""

### Check for duplicates

print('\n\n=========== Duplicate rows in the dataset =============\n\n')

duplicates_df2=athlete_df[athlete_df.duplicated(keep='first')]
print('\n-->The number of duplicate different rows is {}.'.format(duplicates_df2.shape[0]))

duplicates_df=athlete_df[athlete_df.duplicated(keep=False)]

print('\n--> The total number of duplicate rows is {}.'.format(duplicates_df.shape[0]))
print('\n\n--> The duplicate DataFrame:\n',duplicates_df.head())
print('\nThe number of distinct athletes with duplicated entries is {}'.format(duplicates_df['Name'].value_counts().shape[0]))
print('\n--> The Sports for with there are duplicate rows are\n',duplicates_df['Sport'].value_counts())





## Inspection for the duplicates in each category

print('\n\n=========== Duplicate rows in Equestrianism, Sailing and Cycling =============\n\n')
print(duplicates_df[duplicates_df['Sport']=='Equestrianism'])
print(duplicates_df[duplicates_df['Sport']=='Cycling'].head())
print(duplicates_df[duplicates_df['Sport']=='Cycling'] ['Year'].value_counts())
print(duplicates_df[duplicates_df['Sport']=='Sailing'].head())
print(duplicates_df[duplicates_df['Sport']=='Sailing']['Year'].value_counts())



print('\n\n=========== Duplicate rows in Art Competition with medals awarded =============\n\n')
print(duplicates_df[duplicates_df['Sport']=='Art Competitions']['Year'].value_counts())
print(athlete_df[athlete_df["Sport"]=='Art Competitions']['Year'].value_counts())
print(duplicates_df[(duplicates_df["Sport"]=='Art Competitions')  & (duplicates_df['Medal']!='no_medal')])


## Removal of duplicates

athlete_no_art_df=athlete_df.drop(athlete_df[athlete_df.Sport=='Art Competitions'].index)

athlete_no_art_df.drop_duplicates(inplace=True)


athlete_df.drop(athlete_df[(athlete_df.Sport!='Art Competitions') & (athlete_df.duplicated())].index,axis=0,inplace=True)


print('\n\n============ Check if duplicates persist===========\n\n')
print(athlete_df[(athlete_df.duplicated(keep=False))&(athlete_df.Sport!='Art Competitions')])



### Create a new profile after the initial changes

try:
    fn=open("OH_processed.html")
    fn.close()
except:
    print("Creating a profile analysis of the dataset")
    profile = athlete_df.profile_report(title='Olympic History Profiling Report')
    profile.to_file(output_file="OH_processed.html")

try:
    fn=open("OH_no_art_processed.html")
    fn.close()
except:
    print("Creating a profile analysis of the dataset")
    profile_2 = athlete_no_art_df.profile_report(title='Olympic History Profiling Report without Art Competitions')
    profile_2.to_file(output_file="OH_no_art_processed.html")

### Dataset for medalists

medal_df=athlete_df[athlete_df['Medal']!='no_medal']

gold_df=athlete_df[athlete_df['Medal']=='Gold']

silver_df=athlete_df[athlete_df['Medal']=='Silver']

bronze_df=athlete_df[athlete_df['Medal']=='Bronze']

print('\n\n============ Medalists Dataset ===========\n\n')
print(medal_df.head())
print(medal_df.shape)

print('\n\n============ Gold Medalists Dataset ===========\n\n')
print(gold_df.head())
print(gold_df.shape)

print('\n\n============ Silver Medalists Dataset ===========\n\n')
print(silver_df.head())
print(silver_df.shape)

print('\n\n============ Bronze Medalists Dataset ===========\n\n')
print(bronze_df.head())
print(bronze_df.shape)


"""

Creating visual analysis

"""

print("\n\n ==========================================\n Starting Visualization and Analysis\n ========================================== \n\n")
numeric=['Age','Height','Weight']
### Boxplots for numeric variables

print('\n ======== Creating Boxplots ========= \n')

fig, ax_0 = plt.subplots(1,3)
for i in range(len(numeric)):
    sns.boxplot(data=athlete_df[numeric[i]],ax=ax_0[i])
    ax_0[i].title.set_text(numeric[i])
plt.suptitle('Boxplots for numeric features')
plt.savefig('Figures/Boxplot_numeric.pdf')
plt.close()

fig, ax_0 = plt.subplots(1,3)
for i in range(len(numeric)):
    sns.boxplot(data=athlete_df,y=numeric[i],x='Medal',ax=ax_0[i])
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.suptitle('Boxplots for numeric features split by medals')
plt.savefig('Figures/Boxplot_numeric_medals.pdf')
plt.close()

age50=athlete_df[athlete_df.Age>=50].Sport.value_counts()
age50_gold=gold_df[gold_df.Age>=50].Sport.value_counts()
age50_silver=silver_df[silver_df.Age>=50].Sport.value_counts()
age50_bronze=bronze_df[bronze_df.Age>=50].Sport.value_counts()

print('\n ======== Creating Countplots ========= \n')

sns.catplot(data=athlete_df[(athlete_df.Age>=50)&(athlete_df.Medal.isin(['Gold','Silver','Bronze']))&(athlete_df.Sport.isin(['Art Competitions','Shooting','Equestrianism']))],x='Age',row='Medal',hue='Sport',kind='count')
#plt.title('Countplot for the number of medals on the three sports with the most 50+ years old participants')
plt.savefig('Figures/Count_Sport_medal_50.pdf')
plt.close()

sns.countplot(data=athlete_df[(athlete_df.Age>=50)&(athlete_df.Medal.isin(['Gold','Silver','Bronze']))&(athlete_df.Sport.isin(['Art Competitions','Shooting','Equestrianism','Sailing','Archery']))],x='Sport',palette='muted')
plt.savefig('Figures/Medal_count_50.pdf')
plt.close()

sns.countplot(data=athlete_df[(athlete_df.Age>=50)&(athlete_df.Medal.isin(['Gold']))&(athlete_df.Sport.isin(['Art Competitions','Shooting','Equestrianism','Sailing','Archery']))],x='Sport',palette='muted')
plt.savefig('Figures/Gold_medal_count_50.pdf')
plt.close()

print(athlete_df[(athlete_df.Age>=50)&(athlete_df.Sport.isin(['Athletics','Wrestling']))])

print(athlete_df[(athlete_df.Age<=15)]['Sport'].value_counts())
print(athlete_df[(athlete_df.Age<=15)]['Sport'].value_counts().sum())

sns.catplot(data=athlete_df[(athlete_df.Age<=15)],x='Medal',hue='Sex',kind='count',palette='muted')
plt.savefig('Figures/Countplot_under_15.pdf')
plt.close()


print('\n ======== Creating Heatmap ========= \n')

corr=athlete_df[['Height','Weight','Age']].corr(method='pearson')
sns.heatmap(data=corr,cmap='PuBu',annot=True)
plt.title('Correlation between numeric variables')
plt.savefig('Figures/Corr_heatmap.pdf')
plt.close()

print('\n ======== Creating Time Series ========= \n')

ax=sns.countplot(data=athlete_df,x='Year',palette='muted')
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.title.set_text('Number of participants')
plt.savefig('Figures/Participants_year.pdf')
plt.close()

ax=sns.countplot(data=athlete_df,x='Year',hue='Season',palette='muted')
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.title.set_text('Number of participants')
plt.savefig('Figures/Participants_year_season.pdf')
plt.close()

ax_1=sns.countplot(data=athlete_df,x='Year',hue='Sex')
ax_1.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax_1.set(xlabel='Year',ylabel='Numer of participants',title='Number of participants in the Olympics each Year, by Sex')
plt.savefig('Figures/Participants_year_sex.pdf')
plt.close()

ax_2=sns.countplot(data=athlete_df[athlete_df.Medal.isin(['Gold'])],x='Year',hue='Sex')
ax_2.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.suptitle('Number of gold medals in the Olympics each Year, by Sex')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('Figures/Gold_year_sex.pdf')
plt.close()

ax_2=sns.countplot(data=athlete_df[athlete_df.Medal.isin(['Silver'])],x='Year',hue='Sex')
ax_2.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.suptitle('Number of silver medals in the Olympics each Year, by Sex')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('Figures/Silver_year_sex.pdf')
plt.close()

ax_2=sns.countplot(data=athlete_df[athlete_df.Medal.isin(['Bronze'])],x='Year',hue='Sex')
ax_2.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.suptitle('Number of bronze medals in the Olympics each Year, by Sex')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('Figures/Bronze_year_sex.pdf')
plt.close()


print('\n ======== Creating Continent Analysis ========= \n')

## General area plot preparation
years=list(set(athlete_df['Year'].tolist()))
years.sort()
year_continent_temp=athlete_df.groupby(['Year','Continent'])['ID'].size().reset_index(name='count')
year_continent_df=pd.DataFrame(columns=['Year'])
year_continent_df['Year']=years

print('\n\n-->> Creating new yearly continent dataframe\n\n')
continents=['EU','NAm','SA','AS','AF','OC','Multi','UNK']

for i in continents:
    year_continent_df=pd.merge(year_continent_df,year_continent_temp[year_continent_temp['Continent']==i][['Year','count']].reset_index(),on='Year',how='left')
    year_continent_df.drop('index',axis=1,inplace=True)
    year_continent_df.rename(columns={'count':i},inplace=True)
year_continent_df.fillna(value=0,inplace=True)

print(year_continent_df)

sns.set()
year_continent_df.drop(['Multi','UNK'],axis=1).plot.area(x='Year')
plt.ylabel('Number of participants')
plt.savefig('Figures/Continent_area.pdf')
plt.close()

print('\n\n--> Checking UNK and Multi Continent Athletes')
print(athlete_df[athlete_df.Continent.isin(['Multi','UNK'])]['NOC'].value_counts())
print(athlete_df[athlete_df.NOC.isin(['ROT'])]['Year'].value_counts())
print(athlete_df[athlete_df.NOC.isin(['IOA'])]['Year'].value_counts())

print(athlete_df[athlete_df.Continent.isin(['UNK'])])


## Medal area plot preparation

year_continent_medal_temp=medal_df.groupby(['Year','Continent'])['ID'].size().reset_index(name='count')
year_continent_medal_df=pd.DataFrame(columns=['Year'])
year_continent_medal_df['Year']=years

print('\n\n-->> Creating new yearly continent medals dataframe\n\n')
continents=['EU','NAm','SA','AS','AF','OC','Multi','UNK']

for i in continents: 
    year_continent_medal_df=pd.merge(year_continent_medal_df,year_continent_medal_temp[year_continent_medal_temp['Continent']==i][['Year','count']].reset_index(),on='Year',how='left')
    year_continent_medal_df.drop('index',axis=1,inplace=True)
    year_continent_medal_df.rename(columns={'count':i},inplace=True)
year_continent_medal_df.fillna(value=0,inplace=True)

print(year_continent_medal_df)

sns.set()
year_continent_medal_df.drop(['Multi','UNK'],axis=1).plot.area(x='Year')
plt.ylabel('Number of medals')
plt.savefig('Figures/Continent_area_medals.pdf')
plt.close()

medal_proportion=[year_continent_medal_df[i].sum()/year_continent_df[i].sum() for i in continents]
print('\n\n--> Medal proportion by Continent:\n\n')
print(pd.DataFrame(medal_proportion,index=continents,columns=['Medal Proportion']))

print('\n ======== Creating Country Analysis ========= \n')

country_participations_summer_df=athlete_df[athlete_df.Season=='Summer']['region'].value_counts()

ax=sns.barplot(x=country_participations_summer_df.head(10).index,y=country_participations_summer_df.head(10).values)
ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
plt.title('Most participations in Summer Games')
plt.savefig('Figures/Most_part_summer_countries.pdf')
plt.close()

country_participations_winter_df=athlete_df[athlete_df.Season=='Winter']['region'].value_counts()

ax=sns.barplot(x=country_participations_winter_df.head(10).index,y=country_participations_winter_df.head(10).values)
ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
plt.title('Most participations in Winter Games')
plt.savefig('Figures/Most_part_winter_countries.pdf')
plt.close()



country_medals_summer_df=medal_df[medal_df.Season=='Summer']['region'].value_counts()

ax=sns.barplot(x=country_medals_summer_df.head(10).index,y=country_medals_summer_df.head(10).values)
ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
plt.title('Most medals in Summer Games')
plt.savefig('Figures/Most_medals_summer_countries.pdf')
plt.close()

country_medals_winter_df=medal_df[medal_df.Season=='Winter']['region'].value_counts()

ax=sns.barplot(x=country_medals_winter_df.head(10).index,y=country_medals_winter_df.head(10).values)
ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
plt.title('Most medals in Winter Games')
plt.savefig('Figures/Most_medals_winter_countries.pdf')
plt.close()



### Histograms for the participants ages

def histograms(dataframe,feature,feature_to_group,classes_to_keep,same_hist=True,show=False,norm=False,distinguish_title=''):
    """

    This function plots histograms for a certain 'feature', making a separate histogram for each class in
    'feature_to_group'. One chooses the 'classes_to_keep' from the classes of 'feature_to_group'.

    If 'same_hist' is True, then all classes will be ploted in the same histogram; if False, then a subplot is created for each class.
    If 'show' is False, the Figure will be save; if True it will additionally be shown as the code runs.

    """
    if feature_to_group==None:
        axs=sns.distplot(a=dataframe[dataframe[feature].isnull()==False][feature],kde=False,norm_hist=norm)
        axs.set(xlabel=feature,ylabel='Count',title='Histogram of participants {}'.format(feature))
        plt.savefig('Figures/{}_hist{}.pdf'.format(feature,distinguish_title))
        if show==True:
            plt.show()
            return
        else:
            plt.close()
            return
    if isinstance(classes_to_keep,list):
        class_list=classes_to_keep
    else:
        class_list=list(dataframe[feature_to_group].value_counts().head(classes_to_keep).index)
    if same_hist==True:
        for i in range(len(class_list)):
            axs=sns.distplot(a=dataframe[(dataframe[feature].isnull()==False)&(athlete_df[feature_to_group]==class_list[i])][feature],kde=False,norm_hist=norm)
        axs.set(xlabel=feature,ylabel='Count',title='Histogram of participants {} by {}'.format(feature,feature_to_group))
        axs.legend(class_list)
        plt.savefig('Figures/{}_hist_by_{}{}.pdf'.format(feature,feature_to_group,distinguish_title))
        if show==True:
            plt.show()
        else:
            plt.close()
    else:
        length=len(class_list)
        if length%4==0:
            n_columns=4
            n_rows=length//4
        else:
            n_columns=4
            n_rows=length//4+1
        fig, axs = plt.subplots(nrows=n_rows,ncols=n_columns)
        for i in range(len(class_list)):
            sns.distplot(a=dataframe[(dataframe[feature].isnull()==False)&(athlete_df[feature_to_group]==class_list[i])][feature],kde=False,norm_hist=norm,ax=axs.flatten()[i])
            axs.flatten()[i].title.set_text(class_list[i])
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.suptitle('Histogram of participants {} by {}'.format(feature,feature_to_group))
        plt.savefig('Figures/{}_hist_by_{}{}.pdf'.format(feature,feature_to_group,distinguish_title))
        if show==True:
            plt.show()
        else:
            plt.close()

print('\n ======== Creating Histograms for Age ========= \n')

## Total

histograms(athlete_df,'Age',None,0)

## By Sex

histograms(athlete_df,'Age','Sex',2,same_hist=True)

## By Continent

histograms(athlete_df,'Age','Continent',7,same_hist=False)

## By major Sports

histograms(athlete_df,'Age','Sport',3,same_hist=True)

## By Medal

histograms(athlete_df,'Age','Medal',4,same_hist=True,norm=True)



### Histograms for the participants Height

print('\n ======== Creating Histograms for Height ========= \n')

## Total

histograms(athlete_df,'Height',None,0)

## By Sex

histograms(athlete_df,'Height','Sex',2,same_hist=True)

## By Continent

histograms(athlete_df,'Height','Continent',7,same_hist=False)

## By major Sports

histograms(athlete_df,'Height','Sport',3,same_hist=True)

## By Medal

histograms(athlete_df,'Height','Medal',4,same_hist=True,norm=True)



### Histograms for the participants Weight

print('\n ======== Creating Histograms for Weight ========= \n')

## Total

histograms(athlete_df,'Weight',None,0)

## By Sex

histograms(athlete_df,'Weight','Sex',2,same_hist=True)

## By Continent

histograms(athlete_df,'Weight','Continent',7,same_hist=False,norm=False)
histograms(athlete_df,'Weight','Continent',['EU','SA','NAm'],same_hist=True,norm=True,distinguish_title='_some_normed')

## By major Sports

histograms(athlete_df,'Weight','Sport',3,same_hist=True)

## By Medal

histograms(athlete_df,'Weight','Medal',4,same_hist=True,norm=True)



### Analysis of Height and Weight

print('\n ======== Creating Height-Weight Scatterplot ========= \n')

## Scatterplot
try:
    fn=open('Figures/Scatter_Height_Weight.png')
    fn.close()
except:
    print('\n\n===== Creating Scatterplot of Height and Weight =========\n\n')
    sns.scatterplot(data=athlete_df,x='Height',y='Weight',alpha=0.1,hue='Sex')
    plt.savefig('Figures/Scatter_Height_Weight.png')
    plt.close()

try:
    fn=open('Figures/Scatter_Height_Weight_medals.png')
    fn.close()
except:
    print('\n\n===== Creating Scatterplot of Height and Weight with medal hue =========\n\n')
    sns.scatterplot(data=medal_df,x='Height',y='Weight',alpha=0.3,hue='Medal')
    plt.savefig('Figures/Scatter_Height_Weight_medals.png')
    plt.close()


print('\n ======== Creating Height-Weight KDE jointplot ========= \n')

## Jointplot
try:
    fn=open('Figures/Joint_Height_Weight.pdf')
    fn.close()
except:
    print('\n\n===== Creating Jointplot of Height and Weight =====\n\n')
    sns.jointplot(data=athlete_df,x='Height',y='Weight',kind='kde')
    plt.savefig('Figures/Joint_Height_Weight.pdf')
    plt.close()

print('\n ======== Creating Numeric variables boxplots for each medal class ========= \n')

## Boxplot
fig, ax_2=plt.subplots(nrows=3,ncols=1)
sns.catplot(data=athlete_df,x='Medal',y='Height',kind='box',ax=ax_2.flatten()[0])
sns.catplot(data=athlete_df,x='Medal',y='Weight',kind='box',ax=ax_2.flatten()[1])
sns.catplot(data=athlete_df,x='Medal',y='Age',kind='box',ax=ax_2.flatten()[2])
plt.savefig('Figures/Boxplots_numeric.pdf')
plt.close()




### Age of the participants by Year, by Sex (no Art)

# sns.catplot(data=athlete_no_art_df,hue='Sex', x='Year',y='Age', kind='box')
# plt.show()

# print(athlete_no_art_df[(athlete_no_art_df['Year']<1905)& (athlete_no_art_df['Sex']=='F')])