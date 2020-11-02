import pandas as pd
import numpy as np
import os
from env import host, user, password
import requests
##### from walk through ######

######################### Helper function used in create big_df ########################################

def get_df(name):
    """
    This function takes in the string
    'items', 'stores', or 'sales' and
    returns a df containing all pages and
    creates a .csv file for future use.
    """
    base_url = 'https://python.zach.lol'
    api_url = base_url + '/api/v1/'
    response = requests.get(api_url + name)
    data = response.json()
    
    # create list from 1st page
    my_list = data['payload'][name]
    
    # loop through the pages and add to list
    while data['payload']['next_page'] != None:
        response = requests.get(base_url + data['payload']['next_page'])
        data = response.json()
        my_list.extend(data['payload'][name])
    
    # Create DataFrame from list
    df = pd.DataFrame(my_list)
    
    # Write DataFrame to csv file for future use
    df.to_csv(name + '.csv')
    return df

######################### Params Helper function, can be used in big_df ###############################

def get_df_params(name):
    """
    This function takes in the string
    'items', 'stores', or 'sales' and
    returns a df containing all pages and
    creates a .csv file for future use.
    """
    # Create an empty list names `results`.
    results = []
    
    # Create api_url variable
    api_url = 'https://python.zach.lol/api/v1/'
    
    # Loop through the page parameters until an empty response is returned.
    for i in range(3):
        response =  requests.get(items_url, params = {"page": i+1})    
    
        # We have reached the end of the results
        if len(response.json()) == 0:   
            break
            
        else:
            # Convert my response to a dictionary and store as variable `data`
            data = response.json()
        
            # Add the list of dictionaries to my list
            results.extend(data['payload'][name])
    
    # Create DataFrame from list
    df = pd.DataFrame(results)
    
    # Write DataFrame to csv file for future use
    df.to_csv(name + '.csv')
    
    return df

########################### big_df function  ######################################

def get_store_data():
    """
    This function checks for csv files
    for items, sales, stores, and big_df 
    if there are none, it creates them.
    It returns one big_df of merged dfs.
    """
    # check for csv files or create them
    if os.path.isfile('items.csv'):
        items_df = pd.read_csv('items.csv', index_col=0)
    else:
        items_df = get_df('items')
        
    if os.path.isfile('stores.csv'):
        stores_df = pd.read_csv('stores.csv', index_col=0)
    else:
        stores_df = get_df('stores')
        
    if os.path.isfile('sales.csv'):
        sales_df = pd.read_csv('sales.csv', index_col=0)
    else:
        sales_df = get_df('sales')
        
    if os.path.isfile('big_df.csv'):
        df = pd.read_csv('big_df.csv', index_col=0)
        return df
    else:
        # merge all of the DataFrames into one
        df = pd.merge(sales_df, stores_df, left_on='store', right_on='store_id').drop(columns={'store'})
        df = pd.merge(df, items_df, left_on='item', right_on='item_id').drop(columns={'item'})

        # write merged DateTime df with all data to directory for future use
        df.to_csv('big_df.csv')
        return df

    
############################## opsd energy function  #############################

def germany_daily():
    """
    This function uses or creates the 
    opsd_germany_daily csv and returns a df.
    """
    if os.path.isfile('opsd_germany_daily.csv'):
        df = pd.read_csv('opsd_germany_daily.csv', index_col=0)
    else:
        url = 'https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.csv'
        df = pd.read_csv(url)
        df.to_csv('opsd_germany_daily.csv')
    return df


#################### Acquire ##################

def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'


def read_csv_url_csv(url):
    '''
    This function follows a url to a csv and reads in the csv from the url
    '''
    df = pd.read_csv(url)
    return df


def get_csv_url_data(file_name, url, cached=False):
    '''
    This function reads in data fr if cached == False 
    or if cached == True reads in df from a csv file, returns df
    '''
    if cached or os.path.isfile(file_name) == False:
        df = read_csv_url_csv(url)
    else:
        df = pd.read_csv(file_name, index_col=0)
    return df

def get_sales(base_url, df_name):
    '''
    this needs some work to make more generic, or maybe less generic per Faith's walk through
    '''
    # get first page and make dataframe
    response = requests.get('https://python.zach.lol/api/v1/sales')
    data = response.json()
    data.keys()
    print('max_page: %s' % data['payload']['max_page'])
    print('next_page: %s' % data['payload']['next_page'])
    df_name = pd.DataFrame(data['payload']['sales'])
    # get all pages
    while data['payload']['next_page'] != 'None':
        response = requests.get(base_url + data['payload']['next_page'])
        data = response.json()
        print('max_page: %s' % data['payload']['max_page'])
        print('next_page: %s' % data['payload']['next_page'])
        df_name = pd.concat([df_name, pd.DataFrame(data['payload']['sales'])])
        if data['payload']['next_page'] == None:
            break
    df_name = df_name.reset_index()
    print('full_shape', df_name.shape)
    return df_name

#################### Acquire Mall Customers Data ##################

def new_mall_data():
    '''
    This function reads the mall customer data from the Codeup db into a df,
    write it to a csv file, and returns the df. 
    '''
    sql_query = 'SELECT * FROM customers'
    df = pd.read_sql(sql_query, get_connection('mall_customers'))
    df.to_csv('mall_customers_df.csv')
    return df

def get_mall_data(cached=False):
    '''
    This function reads in mall customer data from Codeup database if cached == False 
    or if cached == True reads in mall customer df from a csv file, returns df
    '''
    if cached or os.path.isfile('mall_customers_df.csv') == False:
        df = new_mall_data()
    else:
        df = pd.read_csv('mall_customers_df.csv', index_col=0)
    return df


############# Zillow Clustering ########  

def cluster_zillow_data():
    '''
    This function reads the  zillow data from the Codeup db into a df,
    write it to a csv file, and returns the df. 
    '''
    sql_query = '''
                select * from properties_2017
                join (select id, logerror, pid, tdate from predictions_2017 pred_2017
                join (SELECT parcelid as pid, Max(transactiondate) as tdate FROM predictions_2017 GROUP BY parcelid) as sq1
                on (pred_2017.parcelid = sq1.pid and pred_2017.transactiondate = sq1.tdate)) as sq2
                on (properties_2017.parcelid = sq2.pid)
                left join airconditioningtype using (airconditioningtypeid)
                left join architecturalstyletype using (architecturalstyletypeid)
                left join buildingclasstype using (buildingclasstypeid)
                left join heatingorsystemtype using (heatingorsystemtypeid)
                left join propertylandusetype using (propertylandusetypeid)
                left join storytype using (storytypeid)
                left join typeconstructiontype using (typeconstructiontypeid)
                left join unique_properties using (parcelid)
                where latitude is not null and longitude is not null
                AND propertylandusetypeid IN ('260', '261', '262', '263', '264', '266', '268', '273', '274', '275', '276', '279');
                '''
    df = pd.read_sql(sql_query, get_connection('zillow'))
    df.to_csv('zillow_cluster_df.csv')
    return df


def get_zillow_cluster_data(cached=False):
    '''
    This function reads in zillow customer data from Codeup database if cached == False 
    or if cached == True reads in mall customer df from a csv file, returns df
    '''
    if cached or os.path.isfile('zillow_cluster_df.csv') == False:
        df = cluster_zillow_data()
    else:
        df = pd.read_csv('zillow_cluster_df.csv', index_col=0)
    return df

########## Zillow Regression Project ##########

def new_zillow_data():
    '''
    This function reads the mall customer data from the Codeup db into a df,
    write it to a csv file, and returns the df. 
    '''
    sql_query = '''
                select *
                from properties_2017
                join predictions_2017 using (parcelid)
                WHERE transactiondate between '2017-05-01' AND '2017-06-30'
                AND propertylandusetypeid IN ('246','247','248','260','261','262','263','264','265','266','268','269','273','274','275','276','279');
                '''
    df = pd.read_sql(sql_query, get_connection('zillow'))
    df.to_csv('zillow_df.csv')
    return df

def get_zillow_data(cached=False):
    '''
    This function reads in zillow customer data from Codeup database if cached == False 
    or if cached == True reads in mall customer df from a csv file, returns df
    '''
    if cached or os.path.isfile('zillow_df.csv') == False:
        df = new_zillow_data()
    else:
        df = pd.read_csv('zillow_df.csv', index_col=0)
    return df

def run():
    print("Acquire: downloading raw data files...")
    # Write code here
    print("Acquire: Completed!")

###################### Acquire Iris Data ######################

def new_iris_data():
    '''
    This function reads the iris data from the Codeup db into a df,
    writes it to a csv file, and returns the df.
    '''
    sql_query = """
                SELECT species_id,
                species_name,
                sepal_length,
                sepal_width,
                petal_length,
                petal_width
                FROM measurements
                JOIN species
                USING(species_id)
                """
    df = pd.read_sql(sql_query, get_connection('iris_db'))
    df.to_csv('iris_df.csv')
    return df

def get_iris_data(cached=False):
    '''
    This function reads in iris data from Codeup database if cached == False
    or if cached == True reads in iris df from a csv file, returns df
    '''
    if cached or os.path.isfile('iris_df.csv') == False:
        df = new_iris_data()
    else:
        df = pd.read_csv('iris_df.csv', index_col=0)
    return df