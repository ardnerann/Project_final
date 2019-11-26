import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import DataTable
import Prediction_tf
import Prediction_co
import pickle
from dash.dependencies import Input, Output, State

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets = external_stylesheets)
paysim = pd.read_csv('ps_tf_co_sample.csv')
# ps_tf = pd.read_csv('ps_tf.csv')
# ps_co = pd.read_csv('ps_co.csv')

app.layout = html.Div(children = [
    html.Center(html.H1('Paysim')),
    dcc.Tabs(value = 'tabs', id = 'tabs-1', children = [
        # Prediction_tf.Tab_Prediction_tf(),
        # Prediction_co.Tab_Prediction_co(),
        DataTable.Tab_DataTable(paysim),
        dcc.Tab(label = 'Fraud Detection for CASH_OUT Transaction', value = 'tab-empat', children = [
            html.Div(children = [
                html.Div(children = [
                    html.P('Total Hour of Transaction:'),
                    dcc.Input(id = 'co-input-hour', type = 'number', value = '')
                ], className = 'col-3'),
                html.Div(children = [
                    html.P('Amount of Transaction:'),
                    dcc.Input(id = 'co-input-trx', type = 'number', value = '')
                ], className = 'col-3'),
                html.Div(children = [
                    html.P('Old Original Account Balance: '),
                    dcc.Input(id = 'co-input-old-amt-org', type = 'number', value = '')
                ], className = 'col-3'),
                html.Div(children = [
                    html.P('New Original Account Balance'),
                    dcc.Input(id = 'co-input-new-amt-org', type = 'number', value = '')
                ], className = 'col-3'),
                html.Div(children = [
                    html.P('Old Destination Account Balance'),
                    dcc.Input(id = 'co-input-old-amt-dest', type = 'number', value = '')
                ], className = 'col-3'),
                html.Div(children = [
                    html.P('New Destination Account Balance'),
                    dcc.Input(id = 'co-input-new-amt-dest', type = 'number', value = '')
                ], className = 'col-3'),
            ],
            className = 'row'),
            html.Div(html.Button('search', id = 'co-predict-fraud'),
            style = {
                'padding' : '25px'
            }),
            html.Div(id = 'co-output', children = [html.Center(html.H1('Please Fill the Value'))])
        ]),
        dcc.Tab(label = 'Fraud Detection for TRANSFER Transaction', value = 'tab-lima', children = [
            html.Div(children = [
                html.Div(children = [
                    html.P('Total Hour of Transaction:'),
                    dcc.Input(id = 'tf-input-hour', type = 'number', value = '')
                ], className = 'col-3'),
                html.Div(children = [
                    html.P('Amount of Transaction:'),
                    dcc.Input(id = 'tf-input-trx', type = 'number', value = '')
                ], className = 'col-3'),
                html.Div(children = [
                    html.P('Old Original Account Balance: '),
                    dcc.Input(id = 'tf-input-old-amt-org', type = 'number', value = '')
                ], className = 'col-3'),
                html.Div(children = [
                    html.P('New Original Account Balance'),
                    dcc.Input(id = 'tf-input-new-amt-org', type = 'number', value = '')
                ], className = 'col-3'),
                html.Div(children = [
                    html.P('Old Destination Account Balance'),
                    dcc.Input(id = 'tf-input-old-amt-dest', type = 'number', value = '')
                ], className = 'col-3'),
                html.Div(children = [
                    html.P('New Destination Account Balance'),
                    dcc.Input(id = 'tf-input-new-amt-dest', type = 'number', value = '')
                ], className = 'col-3'),
            ],
            className = 'row'),
            html.Div(html.Button('search', id = 'tf-predict-fraud'),
            style = {
                'padding' : '25px'
            }),
            html.Div(id = 'tf-output', children = html.Center(html.H1('Please Fill the Value')))
        ])
        ],
    ## Tabs Content Style
    content_style = {
        'fontFamily' : 'Arial',
        'borderBottom' : '1px solid #d6d6d6',
        'borderLeft' : '1px solid #d6d6d6',
        'borderRight' : '1px solid #d6d6d6',
        'padding' : '44px'
        })
    ],
    style = {
        'maxWidth' : '1200px',
        'margin' : '0 auto'
    })

# Callback Datatable
@app.callback(
    Output(component_id = 'data-table', component_property = 'children'),
    [Input(component_id = 'search-table', component_property = 'n_clicks')],
    [State(component_id = 'pokemon-generation', component_property = 'value'),
    State(component_id = 'input-rows', component_property = 'value')])

def create_datatable2(n_clicks, x, y):
    if x != 'All':
        return dash_table.DataTable(
            id = 'Table',
            columns = [{'name' : i, 'id' : i} for i in paysim.columns],
            data = paysim[paysim['Generation'] == x].to_dict('records'),
            page_action = 'native',
            page_current = 0,
            page_size = y
            )
    else:
        return dash_table.DataTable(
            id = 'Table',
            columns = [{'name' : i, 'id' : i} for i in paysim.columns],
            data = paysim.to_dict('records'),
            page_action = 'native',
            page_current = 0,
            page_size = y
            )


#Callback Prediction Transfer
@app.callback(
    Output(component_id = 'tf-output', component_property = 'children'),
    [Input(component_id = 'tf-predict-fraud', component_property = 'n_clicks')],
    [State(component_id = 'tf-input-hour', component_property = 'value'),
    State(component_id = 'tf-input-trx', component_property = 'value'),
    State(component_id = 'tf-input-old-amt-org', component_property = 'value'),
    State(component_id = 'tf-input-new-amt-org', component_property = 'value'),
    State(component_id = 'tf-input-old-amt-dest', component_property = 'value'),
    State(component_id = 'tf-input-new-amt-dest', component_property = 'value')])

def predict(n_clicks, hr, trx, old_amt_org, new_amt_org, old_amt_dest, new_amt_dest):
    if hr == '':
        return html.Center(html.H1('Please Fill the Value'))
    else:
        loadModel = pickle.load(open(r'ps_tf.sav', 'rb'))
        prediction = loadModel.predict(np.array([hr, trx, old_amt_org, new_amt_org, old_amt_dest, new_amt_dest]).reshape(1, -1))[0]
        proba = loadModel.predict_proba(np.array([hr, trx, old_amt_org, new_amt_org, old_amt_dest, new_amt_dest]).reshape(1, -1))[0][prediction]
        out = ['Not-Fraud', 'Fraud']
        return html.Center(html.H1('Your Transaction is {} with probability {}'.format(out[prediction], round(proba, 2))))

#Callback Prediction Cash Out
@app.callback(
    Output(component_id = 'co-output', component_property = 'children'),
    [Input(component_id = 'co-predict-fraud', component_property = 'n_clicks')],
    [State(component_id = 'co-input-hour', component_property = 'value'),
    State(component_id = 'co-input-trx', component_property = 'value'),
    State(component_id = 'co-input-old-amt-org', component_property = 'value'),
    State(component_id = 'co-input-new-amt-org', component_property = 'value'),
    State(component_id = 'co-input-old-amt-dest', component_property = 'value'),
    State(component_id = 'co-input-new-amt-dest', component_property = 'value')])

def predict(n_clicks, hr, trx, old_amt_org, new_amt_org, old_amt_dest, new_amt_dest):
    if hr == '':
        return html.Center(html.H1('Please Fill the Value'))
    else:
        loadModel = pickle.load(open(r'ps_co.sav', 'rb'))
        prediction = loadModel.predict(np.array([hr, trx, old_amt_org, new_amt_org, old_amt_dest, new_amt_dest]).reshape(1, -1))[0]
        proba = loadModel.predict_proba(np.array([hr, trx, old_amt_org, new_amt_org, old_amt_dest, new_amt_dest]).reshape(1, -1))[0][prediction]
        out = ['Not-Fraud', 'Fraud']
        return html.Center(html.H1('Your Transaction is {} with probability {}'.format(out[prediction], round(proba, 2))))

if __name__ == '__main__':
    app.run_server(debug=True)