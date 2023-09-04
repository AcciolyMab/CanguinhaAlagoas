import requests
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
    print(f"Accessing geolocation: latitude: {global_latitude}, longitude: {global_longitude}") # Log de depuração
    if global_latitude is None or global_longitude is None:
        return {"error": "Geolocation not set. Please set it first by calling /set_geolocation."}

    # Recebendo os dados JSON da solicitação
    response_list = await request.json()

    # Crie um DataFrame usando a função create_dataframe
    df = create_dataframe(response_list)  # Adicione esta linha


    product_dict_json, distance_matrix, cnpj_to_index_json, category_info=create_entries_for_solver(df, CATEGORIAS)


    simplified_product_dict = product_to_jsonDict(product_dict_json)


    model, x_vars, y, z, u, result_status, tempo, result_value, selected_items = pcc_Uncapacited_tour(distance_matrix, simplified_product_dict)

    optimization_result = extract_results_and_route(selected_items, simplified_product_dict, distance_matrix, x_vars, result_value)


    '''m = folium.Map(location=[global_latitude, global_longitude], zoom_start=13)

    # Adicione marcadores para os locais na rota
    for location in optimization_result['route']:
        folium.Marker(location).add_to(m)

    # Salve o mapa como um arquivo HTML na pasta templates
    map_file_path = "templates/map_file.html"
    m.save(map_file_path)'''

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

def create_entries_for_solver(df, categorias):

    category_info = {}

    depot_coordinates = (-9.65697, -35.70436)
    product_dict = defaultdict(lambda: defaultdict(list))

    unique_locations = df[['LAT', 'LONG']].drop_duplicates().values.tolist()
    locations = [tuple(loc) for loc in unique_locations]
    locations.insert(0, depot_coordinates)

    cnpj_to_index = {str(coordinates): index for index, coordinates in enumerate(locations)}

    n_markets = len(locations)
    distance_matrix = np.zeros((n_markets, n_markets), dtype=np.float64)

    for i in range(n_markets):
        for j in range(n_markets):
            if i != j:
                dist = geodesic(locations[i], locations[j]).kilometers
                fuel_cost = (dist / 9.5) * 5.98
                distance_matrix[i][j] = dist + fuel_cost

    for index, row in df.iterrows():
        if row['VALOR'] != 0:
            barcode = int(row['CODIGO_BARRAS'])
            cnpj = int(row['CNPJ'])
            mercado = str(row['MERCADO'])
            coordinates = (row['LAT'], row['LONG'])
            endereco = row.get('ENDERECO', '')
            mercado_index = cnpj_to_index[str(coordinates)]

            total_cost = float(row['VALOR']) * distance_matrix[mercado_index][0]

            for categoria_id, categoria_info in categorias.items():
                if barcode in categoria_info['barcode']:
                    categoria_nome = categoria_info['nome']

                    product_dict[categoria_id][barcode].append({
                        'valor': row['VALOR'],
                        'cnpj': cnpj,
                        'mercado': mercado,
                        'coordinates': coordinates,
                        'endereco': endereco,
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

    product_dict_json = json.dumps(product_dict)
    cnpj_to_index_json = json.dumps(cnpj_to_index)

    return product_dict_json, distance_matrix, cnpj_to_index_json, category_info

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

def pcc_Uncapacited_tour(distance_matrix, simplified_product_dict):
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

    # Coletar e retornar resultados

    tempo = np.round(execution_time, 2)
    result_status = model.status
    result_value = model.objective.value()
    selected_items = [(v.name, v.varValue) for v in model.variables() if v.varValue > 0]

    return model, x, y, z, u, result_status, tempo, result_value, selected_items

def extract_results_and_route(selected_items, product_dict, distance_matrix, x_vars, result_value):
    """
    Função para extrair resultados e rota a partir das variáveis do modelo.
    """
    result_dict = {
        'total_cost_products': 0,
        'products_by_market': defaultdict(list),
        'market_subtotals': defaultdict(float),
        'route_cost': 0,
        'minimum_route_cost': 0  # Novo campo para o custo mínimo da rota
    }
    product_name = ""  # ou algum valor padrão
    product_cost = 0  # ou algum valor padrão

    # Identificar os mercados selecionados pela variável y
    selected_markets = [int(item[0].split('_')[1]) for item in selected_items if item[0].startswith('y_')]

    # Extração de produtos e custos
    for item in selected_items:
        if item[0].startswith('z_'):
            z_values = item[0][2:-1].replace("(", "").replace(")", "").replace("_", "").replace("'", "").split(",")
            i, category, barcode, cnpj = int(z_values[0]), z_values[1], z_values[2], int(z_values[3])

            # Encontrar as informações do produto
            product_info = next((item for item in product_dict.get(category, {}).get(barcode, []) if item['cnpj'] == cnpj), None)


            if product_info is not None:
                mercado = f"{cnpj} - {product_info['mercado']}"
                product_name = product_info['categoria_nome']
                product_cost = product_info['valor']
            else:
                mercado = f"{cnpj} - Mercado desconhecido"

            result_dict['products_by_market'][mercado].append((product_name, np.round(product_cost, 2)))
            result_dict['market_subtotals'][mercado] += product_cost
            result_dict['total_cost_products'] += product_cost

    # Ordenar os mercados pela sequência em que os produtos foram selecionados
    ordered_markets = []
    for cnpj in selected_markets:
        for category in product_dict:
            for barcode in product_dict[category]:
                if isinstance(product_dict[category][barcode], list):
                    if any(item['cnpj'] == cnpj for item in product_dict[category][barcode]):
                        mercado = f"{cnpj} - {product_dict[category][barcode][0]['mercado']}"
                    else:
                        print(f"Tipo inesperado: {type(product_dict[category][barcode])}. Valor: {product_dict[category][barcode]}")
                        if mercado not in ordered_markets:
                            ordered_markets.append(mercado)

    # Inicializar a rota começando e terminando no DEPOT
    # result_dict['route'] = ['DEPOT'] + ordered_markets + ['DEPOT']

    # Calcular o custo da rota e o custo mínimo
    result_dict['route_cost'] = result_value - result_dict['total_cost_products']
    result_dict['minimum_route_cost'] = result_value

    return result_dict

