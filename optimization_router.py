import requests
import json
import numpy as np
import pandas as pd
import folium
import time
import networkx as nx
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from urllib.parse import unquote
from collections import defaultdict
from itertools import combinations
import setup
from docplex.mp.model import Model
from utils import consultarProduto
from fastapi import FastAPI, APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

optimization_router = APIRouter()


templates = Jinja2Templates(directory="templates")

global_latitude = None
global_longitude = None

@optimization_router.get("/set_geolocation")
def set_geolocation(latitude: float, longitude: float):
    global global_latitude, global_longitude
    global_latitude = latitude
    global_longitude = longitude
    return {"message": "Geolocation set successfully"}

@optimization_router.get("/optimization")
def optimization():
    # Você pode usar global_latitude e global_longitude aqui
    return {"latitude": global_latitude, "longitude": global_longitude}


@optimization_router.post("/create_minimal_cost_list/")
async def create_minimal_cost_list(request: Request):
    global global_latitude, global_longitude

    if global_latitude is None or global_longitude is None:
        return {"error": "Geolocation not set. Please set it first by calling /set_geolocation."}

    # Recebendo os dados JSON da solicitação
    response_list = await request.json()

    #print("Dados recebidos:", response_list)

    # Crie um DataFrame usando a função create_dataframe
    df = create_dataframe(response_list)  # Adicione esta linha

    # Crie o dicionário de produtos
    product_dict = create_product_dict(df)  # Altere esta linha

    # Crie as matrizes de distância e preço
    price_matrix, distance_matrix, indices = create_matrices(product_dict)

    # Chame a função de otimização com os parâmetros apropriados
    optimization_result = comprador_viajante(distance_matrix, price_matrix, product_dict)

    m = folium.Map(location=[global_latitude, global_longitude], zoom_start=13)

    # Adicione marcadores para os locais na rota
    for location in optimization_result['route']:
        folium.Marker(location).add_to(m)

    # Salve o mapa como um arquivo HTML na pasta templates
    map_file_path = "templates/map_file.html"
    m.save(map_file_path)

    # Prepare os dados para o template
    data_to_template = {
        "request": request,
        "optimization_result": optimization_result,
        "map_file_path": map_file_path
    }

    # Renderize o template com os dados
    return templates.TemplateResponse("rotaCustoMinimo.html", data_to_template)


def create_dataframe(response_list):
    # Lista para armazenar os dados
    data = []

    # Iterar sobre cada item na lista de resposta
    for item in response_list:
        # Adicionar cada item como uma linha no DataFrame
        data.append({
            "GTIN": item["gtin"],
            "Descricao": item["descricao"],
            "ValorVenda": item["valorVenda"],
            "CNPJ": item["cnpj"],
            "NomeFantasia": item["nomeFantasia"],
            "Logradouro": item["nomeLogradouro"],
            "NumeroImovel": item["numeroImovel"],
            "Bairro": item["bairro"],
            "Latitude": item["latitude"],
            "Longitude": item["longitude"],
        })

    # Criar um DataFrame usando pandas
    df = pd.DataFrame(data)
    # Criando um dicionário com o mapeamento dos nomes antigos e novos das colunas
    column_mapping = {
        "GTIN": "CODIGO_BARRAS",
        "Descricao": "PRODUTO",
        "ValorVenda": "VALOR",
        "CNPJ": "CNPJ",
        "NomeFantasia": "MERCADO",
        "Logradouro": "ENDERECO",
        "NumeroImovel": "NUMERO",
        "Bairro": "BAIRRO",
        "Latitude": "LAT",
        "Longitude": "LONG",
    }

    # Renomeando as colunas usando o método rename
    df.rename(columns=column_mapping, inplace=True)

    return df


def process_response(response):
    # Implemente a lógica para processar e otimizar a resposta conforme necessário
    # Por exemplo, você pode filtrar, reorganizar ou manipular os dados de alguma forma
    optimized_result = response  # Modifique conforme necessário
    return optimized_result

def filter_and_remove_duplicates(df):
    # FILTRO PARA ELIMINAÇÃO DE PRODUTOS SEM GTIN
    df = df[df['CODIGO_BARRAS'] != 0]
    df = df[df['NCM'] != 0]

    # Drop duplicates based on 'CODIGO_BARRAS' and 'NCM'
    df = df.drop_duplicates(subset=['CODIGO_BARRAS', 'NCM'])

    return df

def rank_and_filter_markets(df):
    # Create a unique ID for each market
    df['unique_market_id'] = df['CNPJ'].astype(str) + ' - ' + df['MERCADO']

    # Get a count of unique barcodes for each market
    product_counts = df.groupby('unique_market_id')[['CODIGO_BARRAS']].nunique()

    # Sort the counts in descending order and get the top 5
    ranking_markets = product_counts.sort_values(by=['CODIGO_BARRAS'], ascending=False).head(5)

    # Convert the Series to a DataFrame
    ranking_markets = ranking_markets.reset_index()

    # Get a list of unique_market_ids for the top 5 markets
    ranking_markets = ranking_markets['unique_market_id'].tolist()

    # Filter df to include only the top 5 markets
    df = df[df['unique_market_id'].isin(ranking_markets)]

    df.reset_index(drop=True, inplace=True)

    return df

def create_product_dict(df):
    # Initialize the dictionary with the depot information
    product_dict = {'00000': {
        'Depot': {
            'mercado': 'Depot',
            'produto': 'Depot',
            'endereco': ' ',
            'localização': (-9.65697, -35.70436),
            'valor': 0.0
        }
        # Other relevant details for the depot
    }}

    # Iterate through the rows of the dataframe
    for index, row in df.iterrows():
        codigo_barras = str(row['CODIGO_BARRAS'])
        valor = row['VALOR']
        if valor != 0: # Ignore entries with value 0
            mercado_info = {
                row['CNPJ']: {
                    "mercado": row['MERCADO'],
                    "produto": row['PRODUTO'],
                    'endereco': row['ENDERECO'],
                    'numero': row['NUMERO'],
                    "localização": (row['LAT'], row['LONG']),
                    "valor": valor
                }
            }

            # If the barcode is already in the dictionary, update the entry. Otherwise, add a new entry.
            if codigo_barras in product_dict:
                product_dict[codigo_barras].update(mercado_info)
            else:
                product_dict[codigo_barras] = mercado_info

    return product_dict

def create_matrices(product_dict):
    depot_location = product_dict['00000']['Depot']['localização']
    market_locations = [info['localização'] for markets in product_dict.values() for info in markets.values() if 'localização' in info and info['localização'] != depot_location]
    locations = [depot_location] + market_locations
    n = len(locations) # Total locations including depot
    m = len(product_dict) - 1 # Number of products, excluding the depot
    vehicle_consumption = 9.5 # km per liter
    fuel_price_per_lt = 5.44

    # Create the distance and fuel cost matrix
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                distance = geodesic(locations[i], locations[j]).kilometers
                fuel_cost = distance * (1 / vehicle_consumption) * fuel_price_per_lt
                distance_matrix[i][j] = distance + fuel_cost

    # Create indices for the markets (starting at 1 to match matrix indexing, 0 is for depot)
    market_index = {'Depot': 0}
    idx = 1 # Start at 1 to match matrix indexing, 0 is for depot
    for markets in product_dict.values():
        for market_cnpj in markets.keys():
            if market_cnpj != 'Depot' and market_cnpj not in market_index:
                market_index[market_cnpj] = idx
                idx += 1

    # Create the price matrix (including a row for the depot)
    price_matrix = np.full((n, m), float('inf'))
    for product_idx, (product_code, markets_info) in enumerate([item for item in product_dict.items() if item[0] != '00000']):
        for market_cnpj, product_info in markets_info.items():
            price_matrix[market_index[market_cnpj]][product_idx] = product_info['valor']

    return price_matrix, distance_matrix, market_index

def create_market_index_mapping(product_dict):
    market_index_mapping = []
    for markets in product_dict.values():
        for market_cnpj, details in markets.items():
            if market_cnpj != 'Depot':
                market_index_mapping.append(market_cnpj)
    return market_index_mapping

def calculate_markets_to_buy(solution, z, m, n, market_index_mapping):
    markets_to_buy = {}
    for k in range(m):
        for i in range(1, n):
            if solution.get_value(z[k][i]) == 1:
                market_cnpjs = [cnpj for cnpj, details in market_index_mapping.items() if details['index'] == i]
                if market_cnpjs:
                    market_cnpj = market_cnpjs[0]
                    markets_to_buy[k] = [market_cnpj]
                else:
                    raise ValueError(f"Index {i} not found in market_index_mapping")
    return markets_to_buy

def calculate_products_and_subtotals(markets_to_buy, market_index_mapping, product_dict):
    products_by_market = {}
    market_subtotals = {}

    for product_idx, market_cnpj_list in enumerate(markets_to_buy):
        for market_cnpj in market_cnpj_list:
            barcode = list(product_dict.keys())[product_idx + 1] # Skip the depot
            if market_cnpj in product_dict[barcode]:
                details = product_dict[barcode][market_cnpj]
                product_code = details["produto"]
                product_price = details["valor"]
                if market_cnpj not in products_by_market:
                    products_by_market[market_cnpj] = []
                products_by_market[market_cnpj].append((product_code, product_price))
                if market_cnpj not in market_subtotals:
                    market_subtotals[market_cnpj] = 0
                market_subtotals[market_cnpj] += float(product_price)

    return products_by_market, market_subtotals

def filter_min_cost_products(markets_to_buy, product_dict):
    min_cost_products = {}
    for product_idx, market_cnpj_list in enumerate(markets_to_buy):
        barcode = list(product_dict.keys())[product_idx + 1] # Skip the depot
        min_price = float('inf')
        min_market_cnpj = None
        for market_cnpj in market_cnpj_list:
            if market_cnpj in product_dict[barcode]:
                details = product_dict[barcode][market_cnpj]
                product_price = float(details["valor"])
                if product_price < min_price:
                    min_price = product_price
                    min_market_cnpj = market_cnpj
        if min_market_cnpj is not None:
            min_cost_products[product_idx] = min_market_cnpj
            #print(f"Produto {barcode} tem custo mínimo no mercado {min_market_cnpj} com preço {min_price}")

    return min_cost_products

def process_solution(solution, market_index_mapping, n, m, z, product_dict):
    markets_to_buy = [[] for _ in range(m)]
    for k in range(m):
        for i in range(1, n):  # Exclude the depot
            if solution.get_value(z[k][i]) == 1:
                market_cnpj = market_index_mapping[i - 1]  # Subtract 1 to align with 0-based indexing
                markets_to_buy[k].append(market_cnpj)

    # Filtrar apenas os produtos com custo mínimo
    min_cost_products = filter_min_cost_products(markets_to_buy, product_dict)

    # Ajustar a lista markets_to_buy com base nos produtos filtrados
    for product_idx, min_market_cnpj in min_cost_products.items():
        markets_to_buy[product_idx] = [min_market_cnpj]

    products_by_market, market_subtotals = calculate_products_and_subtotals(markets_to_buy, market_index_mapping, product_dict)
    total_cost_products = sum(subtotal for subtotal in market_subtotals.values())
    return total_cost_products, markets_to_buy

def get_market_name_from_cnpj(market_cnpj, product_dict):
    # Encontrar o nome do mercado que corresponde ao CNPJ fornecido
    for markets_info in product_dict.values():
        if market_cnpj in markets_info:
            return markets_info[market_cnpj]['mercado']
    return None # Retornar None se o CNPJ não for encontrado

def find_route(solution, x, locations):
    n = len(x)
    route = [0]  # Start at the depot
    current_location = 0
    route_coordinates = [locations[0]]  # Add depot coordinates
    while True:
        next_location_found = False
        for j in range(n):
            if solution.get_value(x[current_location][j]) == 1:
                if j == 0:  # If returning to depot, stop
                    return route_coordinates
                route_coordinates.append(locations[j])
                current_location = j
                next_location_found = True
                break
        if not next_location_found:
            break
    return route_coordinates


def print_route(route, market_index_mapping):
    route_with_names = ["Depot"]
    for idx in route[1:]:  # Skip the depot
        market_cnpj = [cnpj for cnpj, details in market_index_mapping.items() if details['index'] == idx]
        if market_cnpj:
            market_name = get_market_name_from_cnpj(
                market_cnpj[0])  # You can write this function to get the market name from the CNPJ
            route_with_names.append(market_name)
        else:
            raise ValueError(f"Index {idx} not found in market_index_mapping")
    route_with_names.append("Depot")  # Add the depot at the end
    route_string = " --> ".join(route_with_names)

def comprador_viajante(distance_matrix, price_matrix, product_dict):
    start_time = time.time()
    n = len(distance_matrix) # Number of vertices, including depot
    m = len(product_dict) - 1 # Number of products, excluding depot
    c = distance_matrix      # Travel cost
    b = np.transpose(price_matrix) # Buying cost

    # Create the model
    model = Model(name="route-and-buying-optimization")

    # Decision Variables
    x = [[model.binary_var(name=f'x[{i},{j}]') for j in range(n)] for i in range(n)]
    y = [model.binary_var(name=f'y[{i}]') for i in range(n)]
    z = [[model.binary_var(name=f'z[{k},{i}]') for i in range(n)] for k in range(m)]
    u = [model.continuous_var(name=f'u[{i}]') for i in range(1, n)] # excluding depot


    # Objective Function
    objective_terms = []
    for i in range(n):
        for j in range(n):
            if i != j:
                market_cost=model.sum(c[i][j] * x[i][j])
                objective_terms.append(market_cost)
                #objective_terms.append(c[i][j] * x[i][j])
    for i in range(n):
        for k in range(m):
            if not np.isinf(b[k][i]):
                product_costs=model.sum(b[k][i] * z[k][i])
                objective_terms.append(product_costs)
                #objective_terms.append(b[k][i] * z[k][i])

    model.minimize(sum(objective_terms))

    # Seção de restrições
    # Restrição 1: A soma das arestas saindo de um vértice i é igual a y[i]
    for i in range(n):
        model.add_constraint(model.sum(x[i][j] for j in range(n) if i != j) == y[i])

    # Restrição 2: Um produto deve ser comprado exatamente uma vez
    for k in range(m):
        model.add_constraint(model.sum(z[k][i] for i in range(n) if not np.isinf(b[k][i])) == 1)

    # Restrição 3: Um produto só pode ser comprado se o mercado correspondente for visitado
    for k in range(m):
        for i in range(n):
            model.add_constraint(z[k][i] <= y[i])

    # Restrição 4: A rota deve iniciar e finalizar no depósito
    model.add_constraint(y[0] == 1)

    # Restrições adicionais conforme a formulação
    for i in range(1, n):
        model.add_constraint(u[i-1] >= 0)
        model.add_constraint(u[i-1] <= n-2)
        for j in range(1, n):
            if i != j:
                model.add_constraint(u[i-1] - u[j-1] + n * x[i][j] <= n-2)

    # Restrição 6
    for i in range(1, n):
        model.add_constraint(u[i-1] >= 0)



    # Solve the model
    solution = model.solve()

    end_time = time.time()

    elapsed_time = end_time - start_time
    result = {}

    if solution:
        market_index_mapping = create_market_index_mapping(product_dict)
        total_cost_products, markets_to_buy = process_solution(solution, market_index_mapping, n, m, z, product_dict)

        # Calculate the total cost of distance
        total_cost_distance = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    total_cost_distance += c[i][j] * solution.get_value(x[i][j])

        # Update the total cost calculation
        total_cost = total_cost_products + total_cost_distance

        result['execution_time'] = elapsed_time
        result['total_cost_products'] = np.round(total_cost_products, 2)
        result['total_cost'] = np.round(total_cost, 2)
        locations = [product_info['localização'] for markets_info in product_dict.values() for product_info in
                     markets_info.values()]
        result['route'] = find_route(solution, x, locations)

        # Produtos comprados em cada mercado e subtotais
        products_by_market, market_subtotals = calculate_products_and_subtotals(markets_to_buy, market_index_mapping, product_dict)

        result['products_by_market'] = {}
        result['market_subtotals'] = {}

        for market_cnpj, products in products_by_market.items():
            market_name = get_market_name_from_cnpj(market_cnpj, product_dict)
            found_products = set()
            for product, price in products:
                found_products.add((product, np.round(float(price), 2)))
            result['products_by_market'][f"{market_cnpj} - {market_name}"] = found_products

        for market_cnpj, subtotal in market_subtotals.items():
            result['market_subtotals'][market_cnpj] = np.round(subtotal, 2)

    return result