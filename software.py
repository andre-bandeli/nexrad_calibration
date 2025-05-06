import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading

import h5py
import numpy as np
import xarray as xr
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
from geopy.distance import geodesic


PRECIP_COEFS = {
    'kdp_exponent': 0.82,
    'kdp_constant': 24.9,
    'zh_constant': 0.017,
    'zh_exponent': 0.72,
    'rhohv_threshold': 0.95,
    'snr_threshold': 20
}

def verificar_arquivo(arquivo_hdf):
    """Verifica a integridade do arquivo HDF5"""
    try:
        if not os.path.exists(arquivo_hdf):
            raise FileNotFoundError(f"Arquivo {arquivo_hdf} não encontrado")
        
        if os.path.getsize(arquivo_hdf) == 0:
            raise ValueError("Arquivo está vazio")
        
        with open(arquivo_hdf, 'rb') as f:
            header = f.read(8)
            if not header.startswith(b'\x89HDF\r\n\x1a\n'):
                raise ValueError("Não é um arquivo HDF5 válido")
        
        return True
    except Exception as e:
        print(f"Erro na verificação do arquivo: {str(e)}")
        return False

def find_dataset(hdf_file, quantity, alternative_names=None):
    """Busca recursiva por datasets com atributo quantity correspondente"""
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset) and 'what' in obj.parent:
            attrs = obj.parent['what'].attrs
            if 'quantity' in attrs:
                try:
                    if attrs['quantity'].decode() == quantity:
                        return name
                except:
                    if attrs['quantity'] == quantity:
                        return name
        return None
    
    result = hdf_file.visititems(visitor)
    if result:
        return result
    
    if alternative_names:
        for alt_name in alternative_names:
            result = hdf_file.visititems(lambda name, obj: visitor(name, obj)) if quantity == alt_name else None
            if result:
                return result
    return None

def carregar_variavel(f, nome, nome_alternativo=None):
    try:
        path = find_dataset(f, nome, [nome_alternativo] if nome_alternativo else None)
        if path is None:
            raise ValueError(f"Variável {nome} não encontrada")
        
        raw_data = f[path][:]
        what_path = path.rsplit('/', 1)[0] + '/what'
        what_group = f[what_path]
        
        gain = what_group.attrs.get('gain', 1.0)
        offset = what_group.attrs.get('offset', 0.0)
        nodata = what_group.attrs.get('nodata', -9999)
        undetect = what_group.attrs.get('undetect', -9999)
        
        calibrated_data = raw_data * gain + offset
        calibrated_data = np.where((raw_data == nodata) | (raw_data == undetect), np.nan, calibrated_data)
        return calibrated_data
    except Exception as e:
        print(f"Erro ao carregar {nome}: {str(e)}")
        return None

def radar_to_geographic_xr(azimuths_1d, ranges_1d, elevation_deg, radar_lat=-22.81388, radar_lon=-47.05611):
    """Converte coordenadas de radar para geográficas"""
    az_rad = np.deg2rad(azimuths_1d)
    el_rad = np.deg2rad(elevation_deg)
    az_grid, range_grid = np.meshgrid(az_rad, ranges_1d * 1000, indexing='ij')
    
    x = range_grid * np.sin(az_grid) * np.cos(el_rad)
    y = range_grid * np.cos(az_grid) * np.cos(el_rad)
    
    dlon = x / (111319.9 * np.cos(np.deg2rad(radar_lat)))
    dlat = y / 111319.9
    
    return radar_lat + dlat, radar_lon + dlon

def aplicar_filtros(radar_data):
    """Aplica filtros de qualidade nos dados de radar"""
    try:
        mask = (
            (radar_data['DBZH'] > PRECIP_COEFS['snr_threshold']) & 
            (radar_data['RHOHV'] > PRECIP_COEFS['rhohv_threshold'])
        )
        return radar_data.where(mask)
    except Exception as e:
        print(f"Erro ao aplicar filtros: {str(e)}")
        return radar_data

def verificar_pre_processamento(radar_data):
    try:
        zh = radar_data['DBZH'].values
        if np.nanmin(zh) > PRECIP_COEFS['snr_threshold']:
            print("Aviso: Dados podem já ter filtro de SNR aplicado")
    except Exception as e:
        print(f"Erro ao verificar pré-processamento: {str(e)}")

def corrigir_atenuacao(zh, phidp, alpha=0.08):
    """Correção simplificada de atenuação usando PHIDP"""
    try:
        delta_zh = alpha * phidp
        return zh + delta_zh
    except Exception as e:
        print(f"Erro na correção de atenuação: {str(e)}")
        return zh

def estimar_precipitacao(zh, zdr, kdp):
    try:
        # R(KDP) = a * KDP^b
        r_kdp = PRECIP_COEFS['kdp_constant'] * np.power(kdp, PRECIP_COEFS['kdp_exponent'])
        
        # R(ZH) = c * 10^(d*ZH)
        r_zh = PRECIP_COEFS['zh_constant'] * np.power(10, PRECIP_COEFS['zh_exponent'] * zh)
        
        return (r_kdp + r_zh) / 2
    except Exception as e:
        print(f"Erro ao estimar precipitação: {str(e)}")
        return np.full_like(zh, np.nan)

def processar_arquivo_radar_completo(arquivo_hdf):
    try:
        print(f"Iniciando processamento do arquivo: {arquivo_hdf}")
        
        if not verificar_arquivo(arquivo_hdf):
            print("Falha na verificação do arquivo")
            return None

        with h5py.File(arquivo_hdf, 'r', libver='latest') as f:
            print("Buscando variáveis no arquivo HDF...")
            var_paths = {
                'DBZH': find_dataset(f, 'DBZH'),
                'ZDR': find_dataset(f, 'ZDR', 'ZDR_CORR'),
                'PHIDP': find_dataset(f, 'PHIDP'),
                'RHOHV': find_dataset(f, 'RHOHV', 'RHO_HV'),
                'DBZV': find_dataset(f, 'DBZV', 'DBZV_CORR')
            }
            
            missing_vars = [k for k, v in var_paths.items() if v is None]
            if missing_vars:
                print(f"Variáveis essenciais não encontradas: {missing_vars}")
                return None
            print("Todas as variáveis essenciais encontradas")

            radar_vars = {}
            for var_name, path in var_paths.items():
                try:
                    print(f"Processando variável: {var_name}")
                    raw_data = f[path][:]
                    what_path = path.rsplit('/', 1)[0] + '/what'
                    what_group = f[what_path]
                    
                    gain = what_group.attrs.get('gain', 1.0)
                    offset = what_group.attrs.get('offset', 0.0)
                    nodata = what_group.attrs.get('nodata', -9999)
                    undetect = what_group.attrs.get('undetect', -9999)
                    
                    calibrated_data = raw_data * gain + offset
                    calibrated_data = np.where((raw_data == nodata) | (raw_data == undetect), np.nan, calibrated_data)
                    radar_vars[var_name] = calibrated_data
                except Exception as e:
                    print(f"Erro ao processar {var_name}: {str(e)}")
                    return None

            print("Calculando KDP...")
            try:
                kdp = np.gradient(radar_vars['PHIDP'], axis=1)
            except Exception as e:
                print(f"Erro no cálculo de KDP: {str(e)}")
                return None

            print("Extraindo metadados...")
            try:
                what_group_main = f['what']
                date = what_group_main.attrs.get('date', b'19700101').decode('utf-8')
                time = what_group_main.attrs.get('time', b'000000').decode('utf-8')
                data_hora = pd.to_datetime(date + time, format='%Y%m%d%H%M%S')
            except Exception as e:
                print(f"Erro ao ler metadados: {str(e)}")
                data_hora = pd.Timestamp.now()

            print("Configurando coordenadas...")
            radar_lat = -22.81388
            radar_lon = -47.05611
            try:
                elevation = f['how'].attrs.get('elangle', 0.5)
            except:
                elevation = 0.5  # Fallback
            num_azimuths = radar_vars['DBZH'].shape[0]
            num_gates = radar_vars['DBZH'].shape[1]
            range_step = 1000
            azimuths = np.linspace(0, 360, num_azimuths, endpoint=False)
            ranges_km = np.arange(range_step, (num_gates + 1) * range_step, range_step) / 1000.0

            latitudes, longitudes = radar_to_geographic_xr(azimuths, ranges_km, elevation)

            print("Extraindo metadados...")
            
            try:
                # Parse date and time from filename
                filename = os.path.basename(arquivo_hdf)
                parts = filename.split('-')
                if len(parts) >= 4:
                    date_str = parts[2]  # e.g., '20250210'
                    time_str = parts[3]  # e.g., '162000'
                    data_hora = pd.to_datetime(date_str + time_str, format='%Y%m%d%H%M%S')
                else:
                    # Fallback to HDF attributes if filename parsing fails
                    what_group_main = f['what']
                    date = what_group_main.attrs.get('date', b'19700101').decode('utf-8')
                    time = what_group_main.attrs.get('time', b'000000').decode('utf-8')
                    data_hora = pd.to_datetime(date + time, format='%Y%m%d%H%M%S')
            except Exception as e:
                print(f"Erro ao ler metadados: {str(e)}")
                data_hora = pd.Timestamp.now()

            print("Criando dataset xarray...")
            radar_data = xr.Dataset(
                {
                    "DBZH": (("azimuth", "range"), radar_vars['DBZH'], {"units": "dBZ"}),
                    "ZDR": (("azimuth", "range"), radar_vars['ZDR'], {"units": "dB"}),
                    "PHIDP": (("azimuth", "range"), radar_vars['PHIDP'], {"units": "degrees"}),
                    "KDP": (("azimuth", "range"), kdp, {"units": "degrees/km"}),
                    "RHOHV": (("azimuth", "range"), radar_vars['RHOHV'], {"units": ""}),
                    "DBZV": (("azimuth", "range"), radar_vars['DBZV'], {"units": "dBZ"}),
                    "latitude": (("azimuth", "range"), latitudes),
                    "longitude": (("azimuth", "range"), longitudes),
                },
                coords={
                    "azimuth": azimuths,
                    "range": ranges_km,
                    "time": data_hora,
                    "radar_latitude": radar_lat,
                    "radar_longitude": radar_lon,
                    "elevation_angle": elevation,
                }
            )
            
            print("Processamento concluído com sucesso!")
            return radar_data

    except Exception as e:
        print(f"ERRO CRÍTICO ao processar {arquivo_hdf}: {str(e)}")
        return None


def calibrar_zh_zdr_kdp(radar_data):
    """ Método de auto-consistência para calibração"""
    try:
        zh = radar_data['DBZH'].values.flatten()
        zdr = radar_data['ZDR'].values.flatten()
        kdp = radar_data['KDP'].values.flatten()
        rhohv = radar_data['RHOHV'].values.flatten()
        
        mask = (
            ~np.isnan(zh) & ~np.isnan(zdr) & ~np.isnan(kdp) & 
            ~np.isnan(rhohv) & 
            (rhohv > PRECIP_COEFS['rhohv_threshold']) & 
            (zh > PRECIP_COEFS['snr_threshold'])
        )
        
        zh = zh[mask]
        zdr = zdr[mask]
        kdp = kdp[mask]
        
        if len(zh) < 1000:
            print(f"Dados insuficientes para calibração ({len(zh)} pontos válidos)")
            return None

        def kdp_model(zh, zdr, a, b, c):
            """Modelo de auto-consistência KDP = a·10^(b·ZH + c·ZDR)"""
            return a * np.power(10, zh*b/10) * np.power(10, zdr*c/10)

        popt, pcov = curve_fit(
            lambda x, a, b, c: kdp_model(x[0], x[1], a, b, c),
            [zh, zdr], kdp,
            bounds=([1e-6, 0.5, -2], [1e-3, 1.5, 0]),
            maxfev=5000
        )
        
        a_opt, b_opt, c_opt = popt
        
        kdp_pred = kdp_model(zh, zdr, a_opt, b_opt, c_opt)
        delta_zh = np.mean(zh - (10*np.log10(kdp_pred) - 10*np.log10(a_opt) - zdr*c_opt)/b_opt)

        light_rain = (zh < 30) & (zh > PRECIP_COEFS['snr_threshold']) & (np.abs(zdr) < 1.0)
        zdr_bias = np.nanmedian(zdr[light_rain]) if np.sum(light_rain) > 100 else 0.0

        return {
            'a': a_opt, 'b': b_opt, 'c': c_opt,
            'zh_bias': delta_zh,
            'zdr_bias': zdr_bias,
            'n_samples': len(zh),
            'rms_error': np.sqrt(np.mean((kdp - kdp_pred)**2)),
            'pcov': pcov
        }

    except Exception as e:
        print(f"Erro na calibração: {str(e)}")
        return None

def aplicar_calibracao(radar_data, params):
    """Aplica parâmetros de calibração com verificação de segurança"""
    if params is None or radar_data is None:
        return radar_data
    
    try:
        radar_data = radar_data.copy()
        
        radar_data['DBZH_calib'] = radar_data['DBZH'] - params['zh_bias']
        radar_data['ZDR_calib'] = radar_data['ZDR'] - params['zdr_bias']
        
        zh_corr = radar_data['DBZH_calib'].values
        zdr_corr = radar_data['ZDR_calib'].values
        
        valid_mask = ~np.isnan(zh_corr) & ~np.isnan(zdr_corr)
        kdp_corr = np.full_like(zh_corr, np.nan)
        
        a, b, c = params['a'], params['b'], params['c']
        kdp_corr[valid_mask] = a * np.power(10, zh_corr[valid_mask]*b/10) * np.power(10, zdr_corr[valid_mask]*c/10)
        
        radar_data['KDP_calib'] = (('azimuth', 'range'), kdp_corr)
        
        precip_calib = estimar_precipitacao(
            zh_corr,
            zdr_corr,
            kdp_corr
        )
        radar_data['PRECIP_calib'] = (("azimuth", "range"), precip_calib)
        
        return radar_data
    except Exception as e:
        print(f"Erro ao aplicar calibração: {str(e)}")
        return radar_data
        
def carregar_dados_estacoes(pasta_estacoes):
    """Carrega e concatena todos os dados das estações meteorológicas"""
    dados_estacoes = []
    
    for arquivo in os.listdir(pasta_estacoes):
        if arquivo.endswith('.xlsx'):
            caminho = os.path.join(pasta_estacoes, arquivo)
            try:
                df = pd.read_excel(caminho)
                df['Estacao'] = arquivo.replace('.xlsx', '')
                
                df['Precipitação'] = pd.to_numeric(df['Precipitação'], errors='coerce')
                
                df = df.dropna(subset=['Precipitação'])
                
                dados_estacoes.append(df)
            except Exception as e:
                print(f"Erro ao carregar {arquivo}: {str(e)}")
    
    if not dados_estacoes:
        raise ValueError("Nenhum dado de estação carregado")
    
    return pd.concat(dados_estacoes, ignore_index=True)
from math import radians, sin, cos, sqrt, asin

def haversine(radar_lon, radar_lat, estacao_lon, estacao_lat):
    # Raio da Terra em km
    R = 6371.0
    
    # Converter para radianos
    lon1, lat1 = np.radians(radar_lon), np.radians(radar_lat)
    lon2, lat2 = np.radians(estacao_lon), np.radians(estacao_lat)
    
    # Diferenças
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    # Fórmula de Haversine vetorizada
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

def comparar_radar_estacoes(radar_data, dados_estacoes, raio_max_km=5):
    """
    Compara dados de radar com observações de estações meteorológicas
    
    Args:
        radar_data: Dataset xarray com dados de radar processados
        dados_estacoes: DataFrame com dados das estações
        raio_max_km: Raio máximo para considerar coincidência (em km)
    
    Returns:
        DataFrame com dados comparados
    """
    # Converter a coluna de data das estações para datetime
    dados_estacoes['Data'] = pd.to_datetime(dados_estacoes['Data da Coleta'])
    
    # Converter para UTC se as estações estão em RMC (fuso horário local)
    # Supondo que RMC está em UTC-3 (por exemplo, horário de Brasília)
    dados_estacoes['Data_UTC'] = dados_estacoes['Data'].dt.tz_localize('America/Sao_Paulo').dt.tz_convert('UTC')
    
    # Extrair tempo do radar (já em UTC)
    radar_time = pd.to_datetime(radar_data.time.values).tz_localize('UTC')
    
    # Filtrar estações com diferença de tempo menor que 5 minutos (300 segundos)
    estacoes_filtradas = dados_estacoes[
        np.abs((dados_estacoes['Data_UTC'] - radar_time).dt.total_seconds()) <= 300
    ].copy()
    
    if estacoes_filtradas.empty:
        print("Nenhuma estação com dados coincidentes no tempo")
        return None

    resultados = []
    
    for _, estacao in estacoes_filtradas.iterrows():
        # Calcular distâncias vetorizado
        distancias = haversine(
            radar_data.longitude.values,
            radar_data.latitude.values,
            estacao['Lon'],
            estacao['Lat']
        )
        
        # Encontrar o ponto mais próximo
        min_dist_idx = np.unravel_index(np.argmin(distancias), distancias.shape)
        min_dist_km = distancias[min_dist_idx]
        
        if min_dist_km > raio_max_km:
            continue 

        # Extrair valores do radar
        zh_val = radar_data.DBZH.values[min_dist_idx]
        zdr_val = radar_data.ZDR.values[min_dist_idx] if 'ZDR' in radar_data else np.nan
        kdp_val = radar_data.KDP.values[min_dist_idx] if 'KDP' in radar_data else np.nan
        precip_radar = radar_data.PRECIP_calib.values[min_dist_idx] if 'PRECIP_calib' in radar_data else np.nan

        resultados.append({
            'Estacao': estacao['Estacao'],
            'Data_Estacao': estacao['Data'],  # Mantém o horário original da estação
            'Data_Estacao_UTC': estacao['Data_UTC'],  # Horário em UTC
            'Data_Radar': radar_time,
            'Lat': estacao['Lat'],
            'Lon': estacao['Lon'],
            'Precip_Estacao': estacao['Precipitação'],
            'Distancia_Radar_km': min_dist_km,
            'DBZH': zh_val,
            'ZDR': zdr_val,
            'KDP': kdp_val,
            'Precip_Radar': precip_radar,
            'Azimuth': radar_data.azimuth.values[min_dist_idx[0]],
            'Range': radar_data.range.values[min_dist_idx[1]]
        })
    
    if not resultados:
        print("Nenhuma estação dentro do raio de coincidência")
        return None
    
    return pd.DataFrame(resultados)

def plotar_comparacao(resultados):
    """Gera gráficos comparando radar e estações"""
    if resultados is None or len(resultados) == 0:
        print("Nenhum dado para plotar")
        return
    
    if resultados['Precip_Estacao'].isna().all() or resultados['Precip_Radar'].isna().all():
        print("Nenhum dado válido de precipitação para plotar")
        return
    
    plt.figure(figsize=(15, 10))
    
    try:
        # Extrair informações de tempo para o título
        radar_time = resultados['Data_Radar'].iloc[0]
        estacao_time = resultados['Data_Estacao'].iloc[0]
        
        # Formatar os tempos para exibição
        radar_time_str = radar_time.strftime('%Y-%m-%d %H:%M:%S UTC')
        estacao_time_str = estacao_time.strftime('%Y-%m-%d %H:%M:%S (Local)')
        
        # Título geral com informações temporais
        plt.suptitle(f"Comparação Radar-Estação\n"
                    f"Tempo Radar: {radar_time_str}\n"
                    f"Tempo Estação: {estacao_time_str}", 
                    y=1.02, fontsize=12)

        plt.subplot(2, 2, 1)
        max_precip = max(resultados['Precip_Estacao'].max(), resultados['Precip_Radar'].max())
        plt.scatter(resultados['Precip_Estacao'], resultados['Precip_Radar'], 
                   c=resultados['Distancia_Radar_km'], cmap='viridis')
        plt.colorbar(label='Distância do radar (km)')
        plt.plot([0, max_precip], [0, max_precip], 'r--')
        plt.xlabel('Precipitação Estação (mm)')
        plt.ylabel('Precipitação Radar (mm/h)')
        plt.title('Precipitação Radar vs Estação')
        
        plt.subplot(2, 2, 2)
        plt.scatter(resultados['DBZH'], resultados['Precip_Estacao'],
                   c=resultados['Distancia_Radar_km'], cmap='viridis')
        plt.colorbar(label='Distância do radar (km)')
        plt.xlabel('Refletividade (dBZ)')
        plt.ylabel('Precipitação Estação (mm)')
        plt.title('Refletividade vs Precipitação Observada')
        
        if 'KDP' in resultados and not resultados['KDP'].isna().all():
            plt.subplot(2, 2, 3)
            plt.scatter(resultados['KDP'], resultados['Precip_Estacao'],
                       c=resultados['Distancia_Radar_km'], cmap='viridis')
            plt.colorbar(label='Distância do radar (km)')
            plt.xlabel('KDP (°/km)')
            plt.ylabel('Precipitação Estação (mm)')
            plt.title('KDP vs Precipitação Observada')

        plt.subplot(2, 2, 4)
        plt.scatter(resultados['Lon'], resultados['Lat'], 
                   c=resultados['Precip_Estacao'], cmap='Blues', s=100)
        plt.colorbar(label='Precipitação Estação (mm)')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Localização das Estações e Precipitação')
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Erro ao gerar gráficos: {str(e)}")
        

def plot_resultados(radar_data, radar_data_calib=None):
    """Gera plots comparativos com tratamento de erros"""
    try:
        plt.figure(figsize=(18, 12))
        
        # Plot ZH
        plt.subplot(2, 3, 1)
        plt.hist(radar_data['DBZH'].values.flatten(), bins=50, alpha=0.5, label='Original')
        if radar_data_calib is not None and 'DBZH_calib' in radar_data_calib:
            plt.hist(radar_data_calib['DBZH_calib'].values.flatten(), bins=50, alpha=0.5, label='Calibrado')
        plt.title('Distribuição de ZH (dBZ)')
        plt.legend()
        
        # Plot ZDR
        plt.subplot(2, 3, 2)
        plt.hist(radar_data['ZDR'].values.flatten(), bins=50, alpha=0.5, label='Original')
        if radar_data_calib is not None and 'ZDR_calib' in radar_data_calib:
            plt.hist(radar_data_calib['ZDR_calib'].values.flatten(), bins=50, alpha=0.5, label='Calibrado')
        plt.title('Distribuição de ZDR (dB)')
        plt.legend()
        
        # Plot RHOHV
        plt.subplot(2, 3, 3)
        plt.hist(radar_data['RHOHV'].values.flatten(), bins=50, alpha=0.5)
        plt.title('Distribuição de RHOHV')

        plt.subplot(2, 3, 5)
        kdp = radar_data['KDP'].values.flatten()
        kdp = kdp[~np.isnan(kdp)]
        plt.hist(kdp, bins=50, alpha=0.5, label='Original')
        if radar_data_calib is not None and 'KDP_calib' in radar_data_calib:
            kdp_calib = radar_data_calib['KDP_calib'].values.flatten()
            kdp_calib = kdp_calib[~np.isnan(kdp_calib)]
            plt.hist(kdp_calib, bins=50, alpha=0.5, label='Calibrado')
        plt.title('Distribuição de KDP (°/km)')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Erro ao gerar plots: {str(e)}")

class RadarProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Processador de Dados de Radar")
        self.root.geometry("800x600")
        
        self.radar_file = None
        self.stations_folder = None
        self.radar_data = None
        self.radar_data_calib = None
        self.results = None
        
        self.create_widgets()
        
    def create_widgets(self):

        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        file_frame = ttk.LabelFrame(main_frame, text="Arquivos de Entrada", padding=10)
        file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(file_frame, text="Selecionar Arquivo HDF", 
                  command=self.select_hdf_file).pack(side=tk.LEFT, padx=5)
        self.hdf_label = ttk.Label(file_frame, text="Nenhum arquivo selecionado")
        self.hdf_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(file_frame, text="Selecionar Pasta de Estações", 
                  command=self.select_stations_folder).pack(side=tk.LEFT, padx=5)
        self.stations_label = ttk.Label(file_frame, text="Nenhuma pasta selecionada")
        self.stations_label.pack(side=tk.LEFT, padx=5)
        
        proc_frame = ttk.Frame(main_frame)
        proc_frame.pack(fill=tk.X, pady=5)
        
        self.process_btn = ttk.Button(proc_frame, text="Processar Dados", 
                                     command=self.start_processing)
        self.process_btn.pack(side=tk.LEFT, padx=5)
        
        self.progress = ttk.Progressbar(proc_frame, mode='indeterminate')
        
        self.status_text = tk.Text(main_frame, height=10, state=tk.DISABLED)
        self.status_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        result_frame = ttk.Frame(main_frame)
        result_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(result_frame, text="Mostrar Gráficos Comparativos", 
                  command=self.show_comparison_plots).pack(side=tk.LEFT, padx=5)
        ttk.Button(result_frame, text="Mostrar Gráficos de Dados", 
                  command=self.show_data_plots).pack(side=tk.LEFT, padx=5)
        ttk.Button(result_frame, text="Salvar Resultados", 
                  command=self.save_results).pack(side=tk.LEFT, padx=5)
    
    def log_status(self, message):
        self.status_text.configure(state=tk.NORMAL)
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.see(tk.END)
        self.status_text.configure(state=tk.DISABLED)
    
    def select_hdf_file(self):
        self.radar_file = filedialog.askopenfilename(
            title="Selecione o arquivo HDF",
            filetypes=(("Arquivos HDF", "*.hdf"), ("Todos os arquivos", "*.*"))
        )
        
        if self.radar_file:
            self.hdf_label.config(text=self.radar_file)
    
    def select_stations_folder(self):
        self.stations_folder = filedialog.askdirectory(
            title="Selecione a pasta com dados das estações")
        if self.stations_folder:
            self.stations_label.config(text=self.stations_folder)
    
    def start_processing(self):
        if not self.radar_file:
            messagebox.showerror("Erro", "Selecione um arquivo HDF primeiro!")
            return
        
        self.process_btn.config(state=tk.DISABLED)
        self.progress.pack(side=tk.LEFT, padx=5)
        self.progress.start()
        
        processing_thread = threading.Thread(target=self.run_processing)
        processing_thread.start()
    
    def run_processing(self):
        try:
            self.log_status("Iniciando processamento...")
            
            self.radar_data = processar_arquivo_radar_completo(self.radar_file)
            if self.radar_data is None:
                raise ValueError("Falha no processamento do arquivo HDF")
            
            self.log_status("Dados do radar processados com sucesso!")
            
            self.log_status("Realizando calibração...")
            calib_params = calibrar_zh_zdr_kdp(self.radar_data)
            if calib_params:
                self.radar_data_calib = aplicar_calibracao(self.radar_data, calib_params)
                self.log_status("Calibração concluída com sucesso!")
            else:
                self.log_status("Calibração não foi possível")
            
            if self.stations_folder:
                try:
                    dados_estacoes = carregar_dados_estacoes(self.stations_folder)
                    self.results = comparar_radar_estacoes(
                        self.radar_data_calib if self.radar_data_calib else self.radar_data,
                        dados_estacoes
                    )
                    if self.results is not None:
                        self.log_status(f"Comparação com {len(self.results)} estações realizada!")
                except Exception as e:
                    self.log_status(f"Erro na comparação com estações: {str(e)}")
            
            self.log_status("Processamento concluído!")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro durante o processamento: {str(e)}")
        finally:
            self.root.after(0, self.finish_processing)
    
    def finish_processing(self):
        self.progress.stop()
        self.progress.pack_forget()
        self.process_btn.config(state=tk.NORMAL)
    
    def show_comparison_plots(self):
        if self.results is None or self.results.empty:
            messagebox.showwarning("Aviso", "Nenhum dado de comparação disponível")
            return
        
        plot_window = tk.Toplevel(self.root)
        plot_window.title("Gráficos Comparativos")
        
        fig = Figure(figsize=(12, 8))
        plotar_comparacao(self.results, fig)
        
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def show_data_plots(self):
        if self.radar_data is None:
            messagebox.showwarning("Aviso", "Processe os dados primeiro")
            return
        
        plot_window = tk.Toplevel(self.root)
        plot_window.title("Gráficos de Dados do Radar")
        
        fig = Figure(figsize=(12, 8))
        plot_resultados(self.radar_data, self.radar_data_calib, fig)
        
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def save_results(self):
        if self.radar_data is None:
            messagebox.showwarning("Aviso", "Nenhum dado para salvar")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".nc",
            filetypes=(("NetCDF files", "*.nc"), ("Todos os arquivos", "*.*")))
        
        if file_path:
            try:
                if self.radar_data_calib is not None:
                    self.radar_data_calib.to_netcdf(file_path)
                else:
                    self.radar_data.to_netcdf(file_path)
                self.log_status(f"Dados salvos em: {file_path}")
            except Exception as e:
                messagebox.showerror("Erro", f"Falha ao salvar arquivo: {str(e)}")

def plotar_comparacao(resultados, fig=None):
    """Gera gráficos comparativos em uma figura matplotlib"""
    if fig is None:
        fig = plt.figure(figsize=(15, 10))
    else:
        fig.clear()
    
    try:

        radar_time = resultados['Data_Radar'].iloc[0]
        estacao_time = resultados['Data_Estacao'].iloc[0]
        
        fig.suptitle(
            f"Comparação Radar-Estação\n"
            f"Tempo Radar: {radar_time.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            f"Tempo Estação: {estacao_time.strftime('%Y-%m-%d %H:%M:%S (Local)')}",
            y=1.02, fontsize=12
        )

        ax1 = fig.add_subplot(221)
        max_precip = max(resultados['Precip_Estacao'].max(), resultados['Precip_Radar'].max())
        scatter1 = ax1.scatter(resultados['Precip_Estacao'], resultados['Precip_Radar'], 
                            c=resultados['Distancia_Radar_km'], cmap='viridis')
        fig.colorbar(scatter1, ax=ax1, label='Distância do radar (km)')
        ax1.plot([0, max_precip], [0, max_precip], 'r--')
        ax1.set_xlabel('Precipitação Estação (mm)')
        ax1.set_ylabel('Precipitação Radar (mm/h)')
        ax1.set_title('Precipitação Radar vs Estação')

        ax2 = fig.add_subplot(222)
        scatter2 = ax2.scatter(resultados['DBZH'], resultados['Precip_Estacao'],
                            c=resultados['Distancia_Radar_km'], cmap='viridis')
        fig.colorbar(scatter2, ax=ax2, label='Distância do radar (km)')
        ax2.set_xlabel('Refletividade (dBZ)')
        ax2.set_ylabel('Precipitação Estação (mm)')
        ax2.set_title('Refletividade vs Precipitação Observada')

        if 'KDP' in resultados and not resultados['KDP'].isna().all():
            ax3 = fig.add_subplot(223)
            scatter3 = ax3.scatter(resultados['KDP'], resultados['Precip_Estacao'],
                                 c=resultados['Distancia_Radar_km'], cmap='viridis')
            fig.colorbar(scatter3, ax=ax3, label='Distância do radar (km)')
            ax3.set_xlabel('KDP (°/km)')
            ax3.set_ylabel('Precipitação Estação (mm)')
            ax3.set_title('KDP vs Precipitação Observada')

        ax4 = fig.add_subplot(224)
        scatter4 = ax4.scatter(resultados['Lon'], resultados['Lat'], 
                            c=resultados['Precip_Estacao'], cmap='Blues', s=100)
        fig.colorbar(scatter4, ax=ax4, label='Precipitação Estação (mm)')
        ax4.set_xlabel('Longitude')
        ax4.set_ylabel('Latitude')
        ax4.set_title('Localização das Estações e Precipitação')

        fig.tight_layout()
        return fig
        
    except Exception as e:
        print(f"Erro ao gerar gráficos: {str(e)}")
        return fig

def plot_resultados(radar_data, radar_data_calib=None, fig=None):
    """Gera plots comparativos com tratamento de erros"""
    if fig is None:
        fig = plt.figure(figsize=(18, 12))
    else:
        fig.clear()
    
    try:
        # Plot ZH
        ax1 = fig.add_subplot(231)
        ax1.hist(radar_data['DBZH'].values.flatten(), bins=50, alpha=0.5, label='Original')
        if radar_data_calib is not None and 'DBZH_calib' in radar_data_calib:
            ax1.hist(radar_data_calib['DBZH_calib'].values.flatten(), bins=50, alpha=0.5, label='Calibrado')
        ax1.set_title('Distribuição de ZH (dBZ)')
        ax1.legend()

        # Plot ZDR
        ax2 = fig.add_subplot(232)
        ax2.hist(radar_data['ZDR'].values.flatten(), bins=50, alpha=0.5, label='Original')
        if radar_data_calib is not None and 'ZDR_calib' in radar_data_calib:
            ax2.hist(radar_data_calib['ZDR_calib'].values.flatten(), bins=50, alpha=0.5, label='Calibrado')
        ax2.set_title('Distribuição de ZDR (dB)')
        ax2.legend()

        # Plot RHOHV
        ax3 = fig.add_subplot(233)
        ax3.hist(radar_data['RHOHV'].values.flatten(), bins=50, alpha=0.5)
        ax3.set_title('Distribuição de RHOHV')

        # Plot KDP
        ax5 = fig.add_subplot(235)
        kdp = radar_data['KDP'].values.flatten()
        kdp = kdp[~np.isnan(kdp)]
        ax5.hist(kdp, bins=50, alpha=0.5, label='Original')
        if radar_data_calib is not None and 'KDP_calib' in radar_data_calib:
            kdp_calib = radar_data_calib['KDP_calib'].values.flatten()
            kdp_calib = kdp_calib[~np.isnan(kdp_calib)]
            ax5.hist(kdp_calib, bins=50, alpha=0.5, label='Calibrado')
        ax5.set_title('Distribuição de KDP (°/km)')
        ax5.legend()

        fig.tight_layout()
        return fig
        
    except Exception as e:
        print(f"Erro ao gerar plots: {str(e)}")
        return fig

if __name__ == "__main__":
    root = tk.Tk()
    app = RadarProcessorApp(root)
    root.mainloop()