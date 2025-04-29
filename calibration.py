import h5py
import numpy as np
import xarray as xr
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os

PRECIP_COEFS = {
    'kdp_exponent': 46,
    'kdp_constant': 20.3,
    'zh_constant': 0.036,
    'zh_exponent': 0.7,
    'rhohv_threshold': 0.99,
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
            elevation = 0.5
            num_azimuths = radar_vars['DBZH'].shape[0]
            num_gates = radar_vars['DBZH'].shape[1]
            range_step = 1000
            azimuths = np.linspace(0, 360, num_azimuths, endpoint=False)
            ranges_km = np.arange(range_step, (num_gates + 1) * range_step, range_step) / 1000.0

            latitudes, longitudes = radar_to_geographic_xr(azimuths, ranges_km, elevation)

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

def comparar_radar_estacoes(radar_data, dados_estacoes, raio_max_km=10):
    """
    Compara dados de radar com observações de estações meteorológicas
    
    Args:
        radar_data: Dataset xarray com dados de radar processados
        dados_estacoes: DataFrame com dados das estações
        raio_max_km: Raio máximo para considerar coincidência (em km)
    
    Returns:
        DataFrame com dados comparados
    """
    dados_estacoes['Data'] = pd.to_datetime(dados_estacoes['Data da Coleta'])
    
    radar_time = pd.to_datetime(radar_data.time.values)
    
    estacoes_filtradas = dados_estacoes[
        np.abs((dados_estacoes['Data'] - radar_time).dt.total_seconds()) <= 300
    ].copy()
    
    if estacoes_filtradas.empty:
        print("Nenhuma estação com dados coincidentes no tempo")
        return None

    resultados = []
    
    for _, estacao in estacoes_filtradas.iterrows():
        dist_azimute = np.sqrt(
            (radar_data.latitude - estacao['Lat'])**2 +
            (radar_data.longitude - estacao['Lon'])**2
        )

        min_dist_idx = np.unravel_index(np.nanargmin(dist_azimute.values), dist_azimute.shape)
        min_dist_km = dist_azimute.values[min_dist_idx] * 111
        
        if min_dist_km > raio_max_km:
            continue 

        zh_val = radar_data.DBZH.values[min_dist_idx]
        zdr_val = radar_data.ZDR.values[min_dist_idx] if 'ZDR' in radar_data else np.nan
        kdp_val = radar_data.KDP.values[min_dist_idx] if 'KDP' in radar_data else np.nan
        precip_radar = radar_data.PRECIP_calib.values[min_dist_idx] if 'PRECIP_calib' in radar_data else np.nan

        resultados.append({
            'Estacao': estacao['Estacao'],
            'Data_Estacao': estacao['Data'],
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
        
        # Plot Precipitação
        # plt.subplot(2, 3, 4)
        # precip = radar_data['PRECIP'].values.flatten()
        # precip = precip[~np.isnan(precip) & (precip > 0)]
        # plt.hist(precip, bins=50, alpha=0.5, label='Original')
        # if radar_data_calib is not None and 'PRECIP_calib' in radar_data_calib:
        #     precip_calib = radar_data_calib['PRECIP_calib'].values.flatten()
        #     precip_calib = precip_calib[~np.isnan(precip_calib) & (precip_calib > 0)]
        #     plt.hist(precip_calib, bins=50, alpha=0.5, label='Calibrado')
        # plt.title('Distribuição de Precipitação (mm/h)')
        # plt.legend()
        
        # Plot KDP
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

def main():
    arquivo_hdf = '202502012020.hdf'
    pasta_estacoes = 'estacoes'
    
    try:
        print("\nCarregando dados das estações...")
        dados_estacoes = carregar_dados_estacoes(pasta_estacoes)
        print(f"Dados de {len(dados_estacoes['Estacao'].unique())} estações carregados")
    except Exception as e:
        print(f"Erro ao carregar dados das estações: {str(e)}")
        dados_estacoes = None

    print(f"\nProcessando arquivo: {arquivo_hdf}")
    radar_data = processar_arquivo_radar_completo(arquivo_hdf)
    
    if radar_data is None:
        print("Não foi possível processar o arquivo.")
        return
    
    output_file_raw = 'radar_processado.nc'
    radar_data.to_netcdf(output_file_raw)
    print(f"\nDados processados brutos salvos em {output_file_raw}")
    
    print("\nRealizando calibração...")
    calib_params = calibrar_zh_zdr_kdp(radar_data)
    
    if calib_params:
        print("\nParâmetros de calibração:")
        for k, v in calib_params.items():
            if k != 'pcov':
                print(f"{k}: {v:.4f}")
        
        radar_data_calib = aplicar_calibracao(radar_data, calib_params)
        
        if radar_data_calib is not None:

            output_file_calib = 'radar_calibrado.nc'
            
            radar_data_calib.attrs['title'] = 'Dados de radar calibrados'
            radar_data_calib.attrs['history'] = f'Calibrado em {pd.Timestamp.now()}'
            radar_data_calib.attrs['calibration_parameters'] = str(calib_params)
            
            radar_data_calib.to_netcdf(output_file_calib)
            print(f"\nDados calibrados salvos em {output_file_calib}")
            
            if dados_estacoes is not None:
                print("\nComparando com dados das estações...")
                resultados = comparar_radar_estacoes(radar_data_calib, dados_estacoes)
                
                if resultados is not None:
                    print("\nResultados da comparação:")
                    print(resultados[['Estacao', 'Precip_Estacao', 'Precip_Radar', 'DBZH', 'Distancia_Radar_km']])
                    
                    resultados.to_csv('comparacao_radar_estacoes.csv', index=False)
                    print("\nResultados salvos em comparacao_radar_estacoes.csv")
                    
                    plotar_comparacao(resultados)

    plot_resultados(radar_data, radar_data_calib if calib_params else None)

if __name__ == "__main__":
    main()