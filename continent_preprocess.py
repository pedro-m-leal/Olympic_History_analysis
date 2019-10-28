#Libraries for Data Visualization

import pandas as pd

df=pd.read_csv('raw_continent.csv')


continents_df=df[['official_name_en','IOC','Continent']]

print(continents_df['Continent'].value_counts())

print(continents_df[continents_df['Continent'].isnull()][['official_name_en','IOC']])


## Manually add the 'Continent' label to the missing countries. All of them are in North America, so NA.

continents_df.loc[continents_df['Continent'].isnull(),'Continent']=['NAm']*len(continents_df.loc[continents_df['Continent'].isnull(),'Continent'])

continents_list=[]
continents_list.append(['Malaysia','MAL','AS'])
continents_list.append(['Marshall Islands','MHL','OC'])
continents_list.append(['Russia','URS','EU'])
continents_list.append(['Trinidad (West Indies Federation)','WIF','NAm'])
continents_list.append(['Bohemia (Czech Republic)','BOH','EU'])
continents_list.append(['(South) Yemen','YMD','AS'])
continents_list.append(['Germany (Saar protectorate)','SAA','EU'])
continents_list.append(['(North) Yemen','YAR','AS'])
continents_list.append(['Zimbabwe (Rhodesia)','RHO','AF'])
continents_list.append(['Australasia','ANZ','OC'])
continents_list.append(['Unknown','UNK','UNK'])
continents_list.append(['Czechoslovakia','TCH','EU'])
continents_list.append(['West Germany','FRG','EU'])
continents_list.append(['Yugoslavia','YUG','EU'])
continents_list.append(['Crete (Greece)','CRT','EU'])
continents_list.append(['South Sudan','SSD','AF'])
continents_list.append(['Montenegro','MNE','EU'])
continents_list.append(['North Borneo (Malaysia)','NBO','AS'])
continents_list.append(['Newfoundland (Canada)','NFL','NAm'])
continents_list.append(['Kosovo','KOS','EU'])
continents_list.append(['Serbia and Montenegro','SCG','EU'])
continents_list.append(['Individual Olympic Athletes','IOA','Multi'])
continents_list.append(['Unified Team (Russia)','EUN','EU'])
continents_list.append(['Refugee Olympic Team','ROT','Multi'])
continents_list.append(['East Germany','GDR','EU'])
continents_list.append(['United Arab Republic (Syria/Egypt)','UAR','AS'])
continents_list.append(['Vietnam','VNM','AS'])

continents_df2=pd.DataFrame(continents_list,columns=['official_name_en','IOC','Continent'])

print(continents_df2)
print(continents_df.tail())

continents_df=pd.concat([continents_df,continents_df2],ignore_index=True)

continents_df.rename(columns={'official_name_en': 'official_name_en','IOC' : 'NOC','Continent':'Continent'},inplace=True)

print(continents_df.tail())

continents_df.to_csv(r'continents.csv',index=False)