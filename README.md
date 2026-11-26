# K-means sequencial, OpenMP e CUDA

O projeto agrupa registros da base Covertype em três versões do K-means. A versão sequencial usa uma thread, a versão OpenMP usa várias threads na CPU e a versão CUDA roda em uma GPU NVIDIA.

## Executar no Windows

Abra o PowerShell na raiz do projeto. Se você já tem `data\covtype.bin` e os executáveis em `build\`, pode rodar diretamente. Caso algum arquivo esteja faltando, siga os passos de preparação e compilação abaixo.

```powershell
.\build\kmeans_seq.exe .\data\covtype.bin 32 300

$env:OMP_NUM_THREADS = 8
.\build\kmeans_omp.exe .\data\covtype.bin 32 300

.\build\kmeans_cuda.exe .\data\covtype.bin 32 300
```

Os argumentos são o caminho da base binária, o número de clusters (`32` no exemplo) e o limite de iterações (`300`). Para salvar um rótulo por registro, passe o caminho de saída como quarto argumento.

```powershell
.\build\kmeans_seq.exe .\data\covtype.bin 32 300 .\build\rotulos.txt
```

Cada versão imprime as iterações, o SSE (soma dos quadrados das distâncias aos centroides), os tempos de leitura e processamento e o tamanho dos clusters.

## Preparar a base Covertype

Se `data\covtype.bin` ainda não existir, use o Python 3 para preparar a base. O comando baixa o arquivo original caso ele ainda não esteja na pasta `data`.

```powershell
New-Item -ItemType Directory -Force data
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy

if (-not (Test-Path .\data\covtype.data.gz)) {
    curl.exe -fL https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz -o .\data\covtype.data.gz
}

.\.venv\Scripts\python.exe .\scripts\prepare_dataset.py .\data\covtype.data.gz .\data\covtype.bin --drop-last
```

## Compilar

Instale o Visual Studio Build Tools com o compilador C++ x64 e o Windows SDK. Para a versão CUDA, também são necessários o CUDA Toolkit, uma GPU NVIDIA compatível e o driver correspondente. O script procura o `nvcc.exe` em `.toolchain\cuda\bin`, em `$env:CUDA_PATH\bin` e no `PATH`.

```powershell
.\compilar.cmd       # Todas as versões
.\compilar.cmd cpu   # Apenas sequencial e OpenMP
.\compilar.cmd cuda  # Apenas CUDA
```

Os executáveis são gerados em `build\`. A compilação CUDA usa `sm_89` por padrão, a arquitetura da RTX 4060. Se a sua GPU usar outra arquitetura, defina `$env:CUDA_ARCH` antes de compilar.

O [readme.txt](readme.txt) contém os comandos de compilação manual, a descrição do algoritmo e os resultados das medições do trabalho.
