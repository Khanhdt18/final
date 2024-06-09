import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_file
import pickle
import datetime
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import scipy.optimize as sco

app = Flask(__name__)
retailmodel = pickle.load(open('model1.pkl', 'rb'))
corpmodel = pickle.load(open('model2.pkl', 'rb'))
sc_x = pickle.load(open('sc_x.pkl', 'rb'))
sc_y = pickle.load(open('sc_y.pkl', 'rb'))
cubic = pickle.load(open('cubic.pkl', 'rb'))
tx = pickle.load(open('tx.pkl', 'rb'))
ty = pickle.load(open('ty.pkl', 'rb'))
GBRegr = pickle.load(open('GBRegr.pkl', 'rb'))
sales_time_up = pickle.load(open('sales_time_up.pkl', 'rb'))
uz = pickle.load(open('uz.pkl', 'rb'))
zipcode_up = pickle.load(open('zipcode_up.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/index.html')
def home1():
    return render_template('index.html')

@app.route('/predictretail',methods=['POST'])

def predict1():
    def truncate(n, decimals=0):
        multiplier = 10 ** decimals
        return int(n * multiplier) / multiplier
    
    
    # fuction to map zipcode (input from web app) and zipcode_up
    # this is use to prepare input for retail model
    def convert_zipcode(x):
        k=0
        for i in range(len(zipcode_up)):
            if (x== uz[i]):  
                k=i
        return zipcode_up[k]
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    float_features[2]=convert_zipcode(float_features[2])
    final_features = [np.array(float_features)]
    apartment = np.array(final_features).reshape(1, -1)
    apartment_std = sc_x.transform(apartment)
    apartment_cubic_std = cubic.transform(apartment_std)
    cubic_price_pred = retailmodel.predict(apartment_cubic_std)
    prediction=sc_y.inverse_transform(cubic_price_pred)
    output = truncate(prediction, -2)
    
    return render_template('result.html', prediction_text='$ {:,}'.format(output))

@app.route('/predict_api1',methods=['POST'])
def predict_api1():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = sc_y.inverse_transform(
        retailmodel.predict(cubic.transform(
            sc_x.transform(np.array(list(data.values()))).reshape(1, -1))))
    

    output = prediction[0]
    return jsonify(output)



# Model 2 for corporate customers
@app.route('/indexcorp', methods=['GET', 'POST'])
def predict2():
    def truncate(n, decimals=0):
        multiplier = 10 ** decimals
        return int(n * multiplier) / multiplier
    # function to transfer input_excel file into
    # data frame that ready for corporate model
    
    def get_input(df):
        zipcode_up_column = pd.Series([]) 
        for i in range(len(uz)):
            for k in range(len(df)):
                if (df.iloc[:,17][k] == uz[i]):
                    zipcode_up_column[k] = zipcode_up[i]
        df.insert(17, "zipcode_up", zipcode_up_column)
        input_df= df.astype(float)
        return(input_df)
    
    
    def get_prediction(input_df):
        prices = pd.Series([]) 
        for i in range(len(input_df)):
            apartment = np.array(np.append(sales_time_up[0],input_df.iloc[i,:18])).reshape(1, -1)
            apartment_std = tx.transform(apartment)
            price_std = corpmodel.predict(apartment_std)
            prices[i]= ty.inverse_transform(price_std)
        
        input_df.insert(19, "estimated price", prices)
        input_df= input_df.astype(float)
        input_df=input_df.round(1)
        output_df=input_df.drop("zipcode_up",axis=1)
        return(output_df)
    
    if request.method == 'POST':
        df = pd.read_excel(request.files.get('file'))
        input_df = get_input(df) # add zipcode_up column
        output_df= get_prediction(input_df)
        portfolio_value =truncate((output_df.iloc[:,18].sum(axis=0)),-2)
        with pd.ExcelWriter('static/corpvalue.xlsx') as writer:
            output_df.to_excel(writer, sheet_name='Sheet1', engine='xlsxwriter')
        return render_template('resultcorp.html', prediction_text_corp='$ {:,}'.format(portfolio_value))  


# portfolio management
@app.route('/resultportfolio', methods=['GET', 'POST'])
def portfolio_mangement():
    def truncate(n, decimals=0):
        multiplier = 10 ** decimals
        return int(n * multiplier) / multiplier
    # function to transfer input_excel file into
    # data frame that ready for corporate model
    
    def get_input(df):
        zipcode_up_column = pd.Series([]) 
        for i in range(len(uz)):
            for k in range(len(df)):
                if (df.iloc[:,17][k] == uz[i]):
                    zipcode_up_column[k] = zipcode_up[i]
        df.insert(17, "zipcode_up", zipcode_up_column)
        input_df= df.astype(float)
        return(input_df)
    
    
    def get_realestate_prediction(input_df):
        prices = pd.Series([]) 
        realestate_asset_value = pd.Series([]) 
        for k in range(len(sales_time_up)):
            for i in range(len(input_df)):
                apartment = np.array(np.append(sales_time_up[k],input_df.iloc[i,:18])).reshape(1, -1)
                apartment_std = tx.transform(apartment)
                price_std = corpmodel.predict(apartment_std)
                prices[i]= ty.inverse_transform(price_std)
            realestate_asset_value[len(sales_time_up)-k] =truncate(prices.sum(),-2)
        realestate_asset_value = realestate_asset_value.iloc[::-1].reset_index(drop=True)
        return(realestate_asset_value)
    
    if request.method == 'POST':   
        data_file = request.files.get('file')
        
        data_file_xls =pd.ExcelFile(data_file)
        
        df = pd.read_excel(data_file_xls,'Sheet1')
        # real estate list
        
        df1 = pd.read_excel(data_file_xls,'Sheet2')
        # stock symbols
        
        input_df = get_input(df) # add zipcode_up column
        realestate_asset_value = get_realestate_prediction(input_df)
        
        start = datetime.datetime(2014, 5, 1)
        end = datetime.datetime(2015, 5, 31)
        symbols=np.array(df1)
        noa = len(symbols)
        
        data = pd.DataFrame()
        dfk = pd.DataFrame()
        for sym in symbols:
            dfk = web.get_data_yahoo(sym, start, end, interval='m')
            dfk.reset_index(inplace=True)
            dfk.set_index("Date", inplace=True)
            data[sym] = dfk['Close']
        
        data["R E"] = np.nan
        
        for i in range(len(data)):
            data.iloc[i,noa] = realestate_asset_value[i]
        noa = noa+1
        symbols=np.append(symbols, "R E" )
        
        rets = np.log(data / data.shift(1))
        (data / data.iloc[0]*100).plot(figsize=(8, 5))
        plt.title('Normalized prices over time')
        plt.savefig('static/portfolio.png', bbox_inches='tight')
        
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        cax = ax.matshow(rets.corr(), vmin=-1, vmax=1)
        fig.colorbar(cax)
        ticks = np.arange(0,len(symbols),1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(symbols)
        ax.set_yticklabels(symbols)
        plt.title('Correlation matrix ')
        plt.savefig('static/portfolio_corr.png', bbox_inches='tight')
        
        
        weights = np.random.random(noa)
        weights /= np.sum(weights)
        prets = []
        pvols = []
        
        for p in range (1500):
            weights = np.random.random(noa)
            weights /= np.sum(weights)
            prets.append(np.sum(rets.mean() * weights) * 12)
            pvols.append(np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 12, weights))))
        prets = np.array(prets)
        pvols = np.array(pvols)
        
        def statistics(weights):
            weights = np.array(weights)
            pret = np.sum(rets.mean() * weights) * 12
            pvol = np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 12, weights)))
            return np.array([pret, pvol, pret / pvol])
        def min_func_sharpe(weights):
            return -statistics(weights)[2]
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bnds = tuple((0, 1) for x in range(noa))
        opts = sco.minimize(min_func_sharpe, noa * [1. / noa,],  
                            method='SLSQP', bounds=bnds, constraints=cons)
        def min_func_variance(weights):
            return statistics(weights)[1]**2
        optv = sco.minimize(min_func_variance, noa * [1. / noa,], method='SLSQP', bounds=bnds,constraints=cons)
        bnds = tuple((0, 1) for x in weights)
        def min_func_port(weights):
            return statistics(weights)[1]
        trets = np.linspace(0.0, 0.5, 5)
        tvols = []
        for tret in trets:
            cons = ({'type': 'eq', 'fun': lambda x: statistics(x)[0] - tret},
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            res = sco.minimize(min_func_port, noa * [1. / noa,], method='SLSQP',
                       bounds=bnds, constraints=cons)
            tvols.append(res['fun'])
        tvols = np.array(tvols)
        plt.figure(figsize=(8, 4))
        plt.scatter(pvols, prets, c=prets / pvols, marker='o')
        # random portfolio composition
        plt.scatter(tvols, trets, c=trets / tvols, marker='x')
        # efficient frontier
        plt.plot(statistics(opts['x'])[1], statistics(opts['x'])[0],
                 'r*', markersize=15.0)
        # portfolio with highest Sharpe ratio
        plt.plot(statistics(optv['x'])[1], statistics(optv['x'])[0],
                 'y*', markersize=15.0)
        # minimum variance portfolio
        plt.grid(True)
        plt.xlabel('expected volatility')
        plt.ylabel('expected return')
        plt.colorbar(label='Sharpe ratio')
        plt.savefig('static/portfolio_opt.png', bbox_inches='tight')
        
        # create asset allowcation files
        min_var_df = pd.DataFrame(columns=symbols) 
        min_var_df.loc[0]  =  optv['x'].round(3) 
        min_var_df['Portfolio return'],min_var_df['Portfolio volatility'], min_var_df['Sharpe ratio']= statistics(optv['x']).round(3)
        min_var_df['Portfolio type'] = 'Min variance'
        
        
        best_shape_df = pd.DataFrame(columns=symbols) 
        best_shape_df.loc[0]  =  opts['x'].round(3) 
        best_shape_df['Portfolio return'],best_shape_df['Portfolio volatility'], best_shape_df['Sharpe ratio']= statistics(opts['x']).round(3)
        best_shape_df['Portfolio type'] = 'Max Sharpe ratio'
      
        asset_allocation = min_var_df
        asset_allocation.loc[1]= best_shape_df.loc[0]
        asset_allocation = asset_allocation.set_index('Portfolio type')
        
        writer = pd.ExcelWriter('static/asset_allocation.xlsx', engine='xlsxwriter')
        asset_allocation.to_excel(writer, sheet_name='Sheet1')
        # set auto-fit for excel file by max column width
        worksheet= writer.sheets['Sheet1'] # Access the Worksheet
        header_list = asset_allocation.columns.values.tolist() # Generate list of headers
        header_length = list()
        for i in range(len(header_list)):
            header_length.append(len(header_list[i]))
        for i in range(len(header_list)+1):
            worksheet.set_column(i, i, max(header_length)) # Set column widths based on len(header)
        writer.save() # Save the excel file
    
        return render_template('resultportfolio.html')


@app.route('/graph1/')

def send_graph_1():
    
    return send_file('static/portfolio.png', attachment_filename="portfolio.png" ,cache_timeout=0)


     
@app.route('/graph2/')

def send_graph_2():
    
    return send_file('static/portfolio_corr.png', attachment_filename="portfolio_corr.png",cache_timeout=0)



@app.route('/graph3/')

def send_graph_3():
    
    return send_file('static/portfolio_opt.png', attachment_filename="portfolio_opt.png",cache_timeout=0)


@app.route('/download3')

def send_result():
    
    return send_file('static/corpvalue.xlsx', attachment_filename="corpvalue.xlsx", as_attachment=True ,cache_timeout=0)

@app.route('/download4')

def download_file():
    
    return send_file('static/house_data.xlsx', attachment_filename="house_data.xlsx", as_attachment=True ,cache_timeout=0)

@app.route('/download5')
def send_assets():
    
    return send_file('static/asset_allocation.xlsx', attachment_filename="asset_allocation.xlsx", as_attachment=True ,cache_timeout=0)

@app.route('/download6')

def download_file_1():
    
    return send_file('static/portfolio_data.xlsx', attachment_filename="portfolio_data.xlsx", as_attachment=True ,cache_timeout=0)

@app.route('/predict_api2',methods=['POST'])
def predict_api2():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = ty.inverse_transform(
        corpmodel.predict(
            tx.transform(np.array(list(data.values())).reshape(1, -1))))
    

    output = prediction[0]
    return jsonify(output)



if __name__ == "__main__":
    app.run(debug=True)
