import requests
import re
import json
import numpy as np
import pandas as pd
import time
import networkx as nx
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from urllib.parse import unquote
from collections import defaultdict
from utils import consultarProduto
from fastapi import FastAPI, APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import googlemaps
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, PULP_CBC_CMD, LpStatus, LpContinuous, LpInteger, LpAffineExpression
from itertools import product
from itertools import chain, combinations
from  categories import CATEGORIAS
import warnings


optimization_router = APIRouter()


templates = Jinja2Templates(directory="templates")

global_latitude = None
global_longitude = None

@optimization_router.get("/set_geolocation")
def set_geolocation(latitude: float, longitude: float):
    global global_latitude, global_longitude
    global_latitude = latitude
    global_longitude = longitude
    print(f"Geolocation set to latitude: {global_latitude}, longitude: {global_longitude}") # Log de depuração
    return {"message": "Geolocation set successfully"}

@optimization_router.get("/optimization")
def optimization():
    # Você pode usar global_latitude e global_longitude aqui
    return {"latitude": global_latitude, "longitude": global_longitude}

@optimization_router.get("/results", response_class=HTMLResponse)
async def get_results(request: Request):
    # Faça a chamada para a função de otimização e obtenha os resultados
    optimization_result = create_minimal_cost_list()

    route = optimization_result['route']
    route_json = json.dumps(route) if route else '[]'

    return templates.TemplateResponse("rotaCustoMinimo.html", {
        "request": request,
        "optimization_result": optimization_result,
        "route_json": route_json
    })

@optimization_router.post("/create_minimal_cost_list/")
async def create_minimal_cost_list(request: Request):
    global global_latitude, global_longitude
    print(f"Accessing geolocation: latitude ACHOU: {global_latitude}, longitude: {global_longitude}") # Log de depuração
    if global_latitude is None or global_longitude is None:
        return {"error": "Geolocation not set. Please set it first by calling /set_geolocation."}

    # Recebendo os dados JSON da solicitação
    response_list = await request.json()

    # Crie um DataFrame usando a função create_dataframe
    df = create_dataframe(response_list)

    # 'categorias' contém as informações das categorias de produtos
    product_dict_json, distance_matrix, cnpj_to_index_json, category_info = create_entries_for_solver(df, CATEGORIAS, global_latitude, global_longitude)

    print(f"Debug - product_dict_json: {product_dict_json}")  # Debug

    # Simplificar o dicionário de produtos para algo mais fácil de usar
    simplified_product_dict = product_to_jsonDict(product_dict_json)

    model, x, y, z, u, result_status, tempo, result_value, selected_items = pcc_Uncapacited_tour(distance_matrix, simplified_product_dict)
    optimization_result = extract_results_and_route(selected_items, simplified_product_dict, distance_matrix, x, result_value, product_dict_json)

    # Verifica se 'products_by_market' existe e converte para dicionário padrão
    if 'products_by_market' in optimization_result:
        optimization_result['products_by_market'] = dict(optimization_result['products_by_market'])

    # Verifica se 'market_subtotals' existe e converte para dicionário padrão
    if 'market_subtotals' in optimization_result:
        optimization_result['market_subtotals'] = dict(optimization_result['market_subtotals'])

    # Prepare os dados para o template
    data_to_template = {
        "request": request,
        "optimization_result": optimization_result
        # "map_file_path": map_file_path
    }

    # Renderize o template com os dados
    return templates.TemplateResponse("rotaCustoMinimo.html", data_to_template)

    # Prepare os dados para o template
    data_to_template = {
        "request": request,
        "optimization_result": optimization_result
        #"map_file_path": map_file_path
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
            "ncm": item["ncm"],
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
        "ncm": "NCM",
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

# 1. Função para criar a matriz de distâncias
def create_distance_matrix(locations):
    if not isinstance(locations, list):
        raise TypeError("Expected 'locations' to be a list")
    n_markets = len(locations)
    distance_matrix = np.zeros((n_markets, n_markets), dtype=np.float64)
    for i in range(n_markets):
        for j in range(n_markets):
            if i != j:
                dist = geodesic(locations[i], locations[j]).kilometers
                fuel_cost = (dist / 9.5) * 5.98
                distance_matrix[i][j] = dist + fuel_cost
    print("Matriz de Distâncias:", distance_matrix)
    return distance_matrix


# 2. Função para mapear CNPJ para índices
def map_cnpj_to_index(locations):
    if not isinstance(locations, list):
        raise TypeError("Expected 'locations' to be a list")
    return {str(coordinates): index for index, coordinates in enumerate(locations)}


# 3. Função para processar os produtos do DataFrame
def process_products(df, categorias, distance_matrix, cnpj_to_index):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected 'df' to be a pandas DataFrame")
    if not isinstance(categorias, dict):
        raise TypeError("Expected 'categorias' to be a dictionary")
    product_dict = defaultdict(lambda: defaultdict(list))
    category_info = {}

    for index, row in df.iterrows():
        if float(row['VALOR']) != 0.0:
            barcode = int(row['CODIGO_BARRAS'])
            cnpj = int(row['CNPJ'])
            mercado = str(row['MERCADO'])
            coordinates = (row['LAT'], row['LONG'])
            mercado_index = cnpj_to_index[str(coordinates)]

            total_cost = float(row['VALOR']) * distance_matrix[mercado_index][0]

            for categoria_id, categoria_info in categorias.items():
                if barcode in categoria_info['barcode']:
                    categoria_nome = categoria_info['nome']
                    product_dict[categoria_id][barcode].append({
                        'valor': float(row['VALOR']),
                        'cnpj': cnpj,
                        'mercado': mercado,
                        'coordinates': coordinates,
                        'distancia': distance_matrix[mercado_index][0],
                        'custo_total': total_cost,
                        'categoria_nome': categoria_nome,
                        'categoria_id': categoria_id
                    })

                    if categoria_id not in category_info:
                        category_info[categoria_id] = {
                            'nome': categoria_nome,
                            'id': categoria_id
                        }

    return product_dict, category_info

# 4. Função principal que utiliza as subfunções acima
def create_entries_for_solver(df, categorias, global_latitude, global_longitude):
    unique_locations = df[['LAT', 'LONG']].drop_duplicates().values.tolist()
    locations = [tuple(loc) for loc in unique_locations]

    depot_coordinates = (global_latitude, global_longitude)
    locations.insert(0, depot_coordinates)

    distance_matrix = create_distance_matrix(locations)
    cnpj_to_index = map_cnpj_to_index(locations)

    product_dict, category_info = process_products(df, categorias, distance_matrix, cnpj_to_index)

    product_dict_json = json.dumps(product_dict)
    cnpj_to_index_json = json.dumps(cnpj_to_index)

    return product_dict_json, distance_matrix, cnpj_to_index_json, category_info


# 5. Função para simplificar o dicionário de produtos
def product_to_jsonDict(product_dict_json):
    product_dict = json.loads(product_dict_json)
    simplified_product_dict = {}
    for category_id, barcodes in product_dict.items():
        for barcode, product_list in barcodes.items():
            for product in product_list:
                key = (category_id, barcode, product['cnpj'])
                simplified_product_dict[key] = {
                    'distancia': product['distancia'],
                    'valor': float(product['valor'])
                }
    return simplified_product_dict


# As funções foram definidas em sequência. Você pode testá-las agora.


def pcc_Uncapacited_tour(distance_matrix, simplified_product_dict):
    print("Matriz de distâncias:", distance_matrix)
    print("Dicionário de produtos simplificado:", simplified_product_dict)

    # Obter categorias únicas e o número total de mercados
    unique_categories = set(key[0] for key in simplified_product_dict.keys())
    n_categories = len(unique_categories)
    n_markets = len(distance_matrix)

    # Obter CNPJs únicos (identificadores de mercado)
    cnjps = set(cnpj for (_, _, cnpj) in simplified_product_dict.keys())

    # Contar o número de produtos em cada categoria
    n_product_dict = defaultdict(int)
    for category, _, _ in simplified_product_dict.keys():
        n_product_dict[category] += 1
    n_product_dict = dict(n_product_dict)

    # Inicializar o modelo
    model = LpProblem("Comprador-Viajante-ajustado", LpMinimize)
    start_time = time.time()
    # Variáveis de decisão
    x = LpVariable.dicts("x", ((i, j) for i in range(n_markets) for j in range(n_markets)), 0, 1, 'Binary')
    y = LpVariable.dicts("y", range(n_markets), 0, 1)
    z = LpVariable.dicts("z", ((i, category, barcode, cnpj) for i in range(n_markets) for (category, barcode, cnpj) in
                               simplified_product_dict.keys()), 0, 1, 'Binary')
    u = LpVariable.dicts("u", range(n_markets), 0, n_markets - 1, LpInteger)

    # Função objetivo para o custo da rota
    route_cost = lpSum(distance_matrix[i][j] * x[i, j] for i in range(n_markets) for j in range(n_markets) if i != j)

    # Função objetivo para o custo do produto
    # Aqui, estou assumindo que 'barcodes' é uma lista de códigos de barras únicos. Se não for esse o caso, você deve ajustar isso.
    barcodes = set(barcode for (_, barcode, _) in simplified_product_dict.keys())
    product_combinations = list(product(range(n_markets), unique_categories, barcodes, cnjps))

    # Função objetivo para o custo do produto
    # Função objetivo para o custo do produto
    category_costs = {
        category: lpSum(
            (simplified_product_dict.get((category, barcode, cnpj), {}).get('valor', 0) +
            simplified_product_dict.get((category, barcode, cnpj), {}).get('distancia', 0)) *
            z[i, category, barcode, cnpj]
            for i, (c, barcode, cnpj) in product(range(n_markets), simplified_product_dict.keys())
            if c == category
        )
        for category in unique_categories
    }

    # A função objetivo agora considera apenas um produto de cada categoria
    product_cost = lpSum(category_costs[category] for category in unique_categories)

    # Função objetivo total
    model += route_cost + product_cost

    # Restrições de seleção de um produto por categoria
    for category in unique_categories:
        model += lpSum(
            z[i, category, barcode, cnpj]
            for i, (c, barcode, cnpj) in product(range(n_markets), simplified_product_dict.keys())
            if c == category
        ) == 1

        # Restrições
    for i in range(n_markets):
        model += lpSum(x[i, j] for j in range(n_markets) if i != j) == y[i]

    for k in unique_categories:
        model += lpSum(z[i, k, barcode, cnpj] for i, (category, barcode, cnpj) in
                       product(range(n_markets), simplified_product_dict.keys()) if category == k) == 1

    for i in range(n_markets):
        for k in unique_categories:
            model += lpSum(z[i, k, barcode, cnpj] for (category, barcode, cnpj) in simplified_product_dict.keys() if
                           category == k) <= y[i]

    # Restrição do depósito
    model += y[0] == 1

    # Restrições de subciclo
    for i in range(1, n_markets):
        for j in range(1, n_markets):
            if i != j:
                model += u[i] - u[j] + n_markets * x[i, j] <= n_markets - 1

    for i in range(1, n_markets):
        model += u[i] >= 0
    # Resolver o modelo
    model.solve()
    end_time = time.time()
    execution_time = (end_time - start_time)

    '''print("Função Objetivo:", model.objective)
    print("Restrições:", model.constraints)'''

    # Coletar e retornar resultados

    tempo = np.round(execution_time, 2)
    result_status = model.status
    result_value = model.objective.value()
    selected_items = [(v.name, v.varValue) for v in model.variables() if v.varValue > 0]

    '''print("X da Questão", x)

    print("Status do Solver:", result_status)
    print("Status do Solver:", result_value)
    print("Variáveis de Decisão:", [(v.name, v.varValue) for v in model.variables() if v.varValue > 0])'''

    return model, x, y, z, u, result_status, tempo, result_value, selected_items

def verify_product_dict_format(product_dict):
    # Verificação do formato do dicionário de produtos
    if not isinstance(product_dict, dict):
        raise ValueError("product_dict deve ser um dicionário.")


def extract_product_info(z_value, product_dict):
    z_values = z_value[2:-1].replace("(", "").replace(")", "").replace("_", "").replace("'", "").split(",")
    i, category, barcode, cnpj = int(z_values[0]), z_values[1], z_values[2], int(z_values[3])

    product_info = next((item for item in product_dict.get(category, {}).get(barcode, []) if item['cnpj'] == cnpj), None)

    if product_info is not None:
        mercado = f"{cnpj} - {product_info['mercado']}"
        product_name = product_info['categoria_nome']
        product_cost = product_info['valor']
    else:
        mercado = f"{cnpj} - Mercado desconhecido"
        warnings.warn(f"Produto desconhecido encontrado: {z_value}")
        product_name, product_cost = "Produto desconhecido", 0



    return mercado, product_name, product_cost

def extract_results_and_route(selected_items, simplified_product_dict, distance_matrix, x_vars, result_value,
                              product_dict_json):
    # Inicialização
    total_cost_products = 0
    subtotais_by_market = {}
    products_by_market = {}

    selected_products = [re.findall(r"z_\(\d+,_'(.+)',_'(.+)',_(.+)\)", item[0]) for item in selected_items if
                         'z_' in item[0]]
    selected_products = [item[0] for item in selected_products if item]

    product_dict = json.loads(product_dict_json)

    for (cat, barcode, cnpj) in selected_products:
        cnpj = int(cnpj)
        if cnpj not in products_by_market:
            products_by_market[cnpj] = []

        if (cat, barcode, cnpj) in simplified_product_dict:
            product_info = simplified_product_dict[(cat, barcode, cnpj)]
            additional_info = next(
                (item for item in product_dict.get(cat, {}).get(barcode, []) if item['cnpj'] == cnpj), None)

            if additional_info is not None:
                product_info['mercado'] = additional_info['mercado']
                product_info['categoria_nome'] = additional_info['categoria_nome']

            products_by_market[cnpj].append(product_info)

    subtotais_by_market = {mercado: sum([prod['valor'] for prod in prods]) for mercado, prods in products_by_market.items()}
    total_cost_products = round(sum(subtotais_by_market.values()), 2)

    # Calculando o custo da rota
    #route_cost = round(sum([distance_matrix[i][j] for (i, j), var in x_vars.items() if var.value == 1.0]), 2)
    # Debugging: Verifique os valores que contribuem para o custo da rota
    contributing_values = [(i, j, distance_matrix[i][j], var.value) for (i, j), var in x_vars.items() if
                           var.value == 1.0]

    minimum_route_cost = round(result_value, 2)
    route_cost = round(minimum_route_cost - total_cost_products, 2)

    return {
        "total_cost_products": total_cost_products,
        "route_cost": route_cost,
        "products_by_market": products_by_market,
        "subtotais_by_market": subtotais_by_market,
        "minimum_route_cost": minimum_route_cost,
    }

