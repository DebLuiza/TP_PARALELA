# Benchmark unificado do K-means Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Gerar e documentar tempos reproduzíveis das versões sequencial, OpenMP e CUDA na mesma máquina.

**Architecture:** Um script PowerShell compila os três programas, valida suas saídas e executa uma rodada de aquecimento mais cinco rodadas medidas por configuração. O script preserva as observações em CSV; a mediana e os speedups calculados contra a mediana sequencial alimentam os cabeçalhos dos códigos e o `readme.txt`.

**Tech Stack:** C11, OpenMP, CUDA C++, GCC/MinGW-w64, NVIDIA CUDA Toolkit 13.5, MSVC Build Tools 2022 e PowerShell 7/Windows PowerShell.

**Spec:** `docs/superpowers/specs/2026-11-22-benchmark-unificado-design.md`

## Global Constraints

- Dataset e parâmetros: `data/covtype.bin`, `K = 32`, `max_iter = 300`.
- Medir somente o laço iterativo do K-means; leitura, inicialização, SSE e transferência inicial para a GPU ficam fora.
- Usar uma execução de aquecimento e cinco execuções medidas por configuração.
- Publicar a mediana e calcular todos os speedups contra a nova mediana sequencial.
- Executar OpenMP com 1, 2, 4, 8, 16 e 32 threads; identificar 32 como sobrescrição da CPU de 16 threads lógicas.
- Não alterar o algoritmo ou reduzir o trabalho para melhorar números.
- A versão sequencial medida deve demandar pelo menos 10 segundos.

## Review Focus

- Compilador CUDA sem `cl.exe`: o bootstrap deve instalar/detectar MSVC antes de invocar `nvcc`.
- `nvcc` recém-instalado ausente do `PATH` do processo atual: os scripts devem localizar o executável pela instalação, sem exigir reinício.
- Resultados funcionalmente divergentes: o benchmark deve parar antes das medições se iterações, SSE ou clusters forem incompatíveis.
- Saída localizada do sistema: o parser deve ler apenas as chaves estáveis impressas pelos próprios executáveis e usar cultura invariável para números.
- Falha ou saída incompleta numa repetição: o script deve encerrar com erro, sem calcular mediana de uma série parcial.

---

### Task 1: Preparar e verificar a toolchain CUDA no Windows

**Files:**
- No repository changes.

**Interfaces:**
- Consumes: Windows 11, `winget`, GPU NVIDIA RTX 4060 Ti e driver 610.88.
- Produces: `cl.exe`/MSVC Build Tools 2022 e `nvcc.exe` do CUDA Toolkit 13.5 utilizáveis no processo de compilação.

- [ ] **Step 1: Confirmar a ausência atual das ferramentas**

Run: `where.exe cl; where.exe nvcc`
Expected: ambos ausentes antes da instalação.

- [ ] **Step 2: Instalar o workload C++ do Visual Studio Build Tools 2022**

Run: `winget install --id Microsoft.VisualStudio.2022.BuildTools --exact --accept-package-agreements --accept-source-agreements --override "--wait --quiet --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"`
Expected: instalação concluída com exit code 0.

- [ ] **Step 3: Instalar o NVIDIA CUDA Toolkit 13.5**

Run: `winget install --id Nvidia.CUDA --exact --accept-package-agreements --accept-source-agreements --silent`
Expected: instalação concluída com exit code 0.

- [ ] **Step 4: Verificar compiladores e GPU**

Run: localizar `VsDevCmd.bat` com `vswhere.exe`, localizar `nvcc.exe` sob `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5\bin`, executar `cl`, `nvcc --version` e `nvidia-smi --query-gpu=name,driver_version --format=csv,noheader`.
Expected: MSVC 19.x, CUDA 13.5 e `NVIDIA GeForce RTX 4060 Ti`.

### Task 2: Tornar a unidade CUDA compilável na toolchain aprovada

**Files:**
- Modify: `kmeans_cuda.cu:35-72`

**Interfaces:**
- Consumes: MSVC C++ host compiler usado por `nvcc`.
- Produces: `static double tempo_atual(void)` baseado em `std::chrono::steady_clock`, somente para medir leitura no host; o laço CUDA continua medido por `cudaEvent`.

- [ ] **Step 1: Executar a compilação CUDA atual para registrar a falha de portabilidade**

Run: em um Developer Command Prompt, `nvcc -O2 -arch=sm_89 -o kmeans_cuda.exe kmeans_cuda.cu`
Expected: FAIL em `clock_gettime`/`CLOCK_MONOTONIC` na compilação host do Windows.

- [ ] **Step 2: Substituir a dependência POSIX do cronômetro host**

Em `kmeans_cuda.cu`, substituir `<time.h>` por `<chrono>` e implementar `tempo_atual()` com `std::chrono::steady_clock::now()` convertido para `std::chrono::duration<double>`.

- [ ] **Step 3: Recompilar CUDA**

Run: em um Developer Command Prompt, `nvcc -O2 -arch=sm_89 -o kmeans_cuda.exe kmeans_cuda.cu`
Expected: PASS e geração de `kmeans_cuda.exe` sem warnings do projeto.

- [ ] **Step 4: Executar um teste funcional CUDA curto**

Run: `.\kmeans_cuda.exe .\data\covtype.bin 32 1`
Expected: exit code 0, `Versao : CUDA`, `Iteracoes : 1` e um `Tempo K-means` numérico.

- [ ] **Step 5: Commit**

```powershell
git add kmeans_cuda.cu
git commit -m "fix: torna cronometro CUDA portavel no Windows"
```

### Task 3: Criar o executor reproduzível de benchmark

**Files:**
- Create: `scripts/run_benchmarks.ps1`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: caminhos opcionais `-Gcc`, `-Nvcc`, `-Dataset`, inteiro `-Repeticoes` (default 5) e executáveis compilados no diretório do projeto.
- Produces: `resultados/benchmark-raw.csv` com colunas `versao,threads,rodada,tempo_kmeans_s,iteracoes,sse,clusters`; exit code não zero diante de compilação, execução ou validação incompleta.

- [ ] **Step 1: Escrever um teste de contrato do script**

Criar no próprio script a opção `-ValidarSomente`; ela compila e executa uma vez cada versão, verifica presença das sete configurações requeridas e falha se qualquer campo esperado não puder ser extraído. Antes da implementação do pipeline, executar a opção.

- [ ] **Step 2: Executar o contrato para verificar a falha inicial**

Run: `powershell -ExecutionPolicy Bypass -File scripts/run_benchmarks.ps1 -ValidarSomente`
Expected: FAIL porque o script/pipeline ainda não existe.

- [ ] **Step 3: Implementar descoberta e compilação das três versões**

O script deve resolver caminhos absolutos; compilar sequencial com `gcc -O2 -std=c11 -Wall`, OpenMP com os mesmos flags mais `-fopenmp`, e CUDA com `nvcc -O2 -arch=sm_89`. Deve parar em qualquer exit code não zero. Acrescentar `resultados/benchmark-raw.csv` ao `.gitignore`, pois o arquivo oficial será adicionado explicitamente após a medição.

- [ ] **Step 4: Implementar execução e parser invariável**

Criar `Invoke-KmeansRun`, que recebe versão e número opcional de threads, configura `OMP_NUM_THREADS` somente para OpenMP, captura stdout e extrai `Iteracoes`, `SSE`, `Tempo K-means` e `Tam. clusters`. Validar que todos os campos existem e converter decimais com `InvariantCulture`.

- [ ] **Step 5: Implementar validação funcional**

Executar sequencial, OpenMP com 1 thread e CUDA uma vez. Exigir exatamente o mesmo número de iterações e lista de tamanhos; aceitar SSE CUDA/OpenMP com erro relativo máximo de `1e-5` contra o sequencial. Falhar com diagnóstico se o contrato não for satisfeito.

- [ ] **Step 6: Implementar aquecimento, cinco repetições e CSV**

Executar uma rodada descartada e `Repeticoes` rodadas gravadas para sequencial, CUDA e cada contagem OpenMP `1,2,4,8,16,32`. Criar `resultados/` se necessário e gravar todas as observações em CSV apenas depois de completar a série.

- [ ] **Step 7: Executar somente a validação**

Run: `powershell -ExecutionPolicy Bypass -File scripts/run_benchmarks.ps1 -ValidarSomente`
Expected: PASS, com as três versões compiladas e validadas.

- [ ] **Step 8: Verificar falha segura com dataset inválido**

Run: `powershell -ExecutionPolicy Bypass -File scripts/run_benchmarks.ps1 -ValidarSomente -Dataset .\data\inexistente.bin`
Expected: FAIL antes de criar ou substituir `resultados/benchmark-raw.csv`.

- [ ] **Step 9: Commit**

```powershell
git add scripts/run_benchmarks.ps1 .gitignore
git commit -m "feat: automatiza benchmark comparavel"
```

### Task 4: Executar, analisar e preservar as medições

**Files:**
- Create: `resultados/benchmark-raw.csv` (gerado pelo script e adicionado com `git add -f`)
- Create: `resultados/benchmark-summary.txt`

**Interfaces:**
- Consumes: CSV completo da Task 3.
- Produces: medianas das oito configurações e speedups contra a mediana sequencial.

- [ ] **Step 1: Reduzir interferência evitável do ambiente**

Fechar apenas processos de benchmark antigos iniciados por este fluxo, registrar o plano de energia atual e confirmar que não há outra carga CUDA de computação. Não encerrar aplicações do usuário.

- [ ] **Step 2: Executar a série completa**

Run: `powershell -ExecutionPolicy Bypass -File scripts/run_benchmarks.ps1 -Repeticoes 5`
Expected: 40 linhas de dados (5 para sequencial, 5 para CUDA e 30 para as seis configurações OpenMP), todas válidas.

- [ ] **Step 3: Calcular medianas e speedups**

Gerar `resultados/benchmark-summary.txt` com ambiente, mediana por configuração, mínimo/máximo e `speedup = mediana_sequencial / mediana_configuracao`, usando cultura invariável.

- [ ] **Step 4: Conferir requisitos e plausibilidade**

Verificar: sequencial >= 10 s; cinco observações por configuração; OpenMP reporta o número solicitado de threads; nenhuma mediana é derivada de execução divergente; 32 threads está marcado como sobrescrição. Investigar, sem maquiar, qualquer anomalia.

- [ ] **Step 5: Preservar os resultados**

```powershell
git add -f resultados/benchmark-raw.csv
git add resultados/benchmark-summary.txt
git commit -m "perf: registra benchmark na mesma maquina"
```

### Task 5: Atualizar documentação e comentários exigidos

**Files:**
- Modify: `kmeans_seq.c:1-27`
- Modify: `kmeans_omp.c:1-33`
- Modify: `kmeans_cuda.cu:1-27`
- Modify: `readme.txt`

**Interfaces:**
- Consumes: `resultados/benchmark-summary.txt` e comandos efetivamente verificados.
- Produces: tabela idêntica, completa e rastreável nos três cabeçalhos e documentação reproduzível no readme.

- [ ] **Step 1: Registrar o estado documental que deve falhar**

Run: `rg -n "______|ainda pendente|RTX 3050|sm_86|25\.939|2\.685" kmeans_seq.c kmeans_omp.c kmeans_cuda.cu readme.txt`
Expected: pelo menos uma ocorrência obsoleta antes da edição.

- [ ] **Step 2: Atualizar os três cabeçalhos**

Inserir máquina, dataset, parâmetros, número de repetições, estatística publicada, tempos medianos e speedups da Task 4. Manter os comentários sobre as mudanças de paralelização já existentes.

- [ ] **Step 3: Atualizar `readme.txt`**

Corrigir ambiente para Windows 11/Ryzen 7 5700X3D/RTX 4060 Ti/CUDA 13.5/sm_89; documentar instalação/compilação/execução verificadas, método da mediana, região cronometrada, tabela final, validação e a sobrescrição de 32 threads. Marcar as três implementações como concluídas.

- [ ] **Step 4: Verificar remoção de dados obsoletos**

Run: `rg -n "______|ainda pendente|RTX 3050|sm_86|25\.939|2\.685" kmeans_seq.c kmeans_omp.c kmeans_cuda.cu readme.txt`
Expected: nenhum resultado.

- [ ] **Step 5: Recompilar e executar validação final**

Run: `powershell -ExecutionPolicy Bypass -File scripts/run_benchmarks.ps1 -ValidarSomente`
Expected: PASS nas três compilações, execuções e comparações funcionais.

- [ ] **Step 6: Conferir consistência das tabelas**

Comparar automaticamente os oito tempos e speedups do summary contra cada cabeçalho e o readme.
Expected: valores idênticos e nenhuma lacuna.

- [ ] **Step 7: Commit**

```powershell
git add kmeans_seq.c kmeans_omp.c kmeans_cuda.cu readme.txt
git commit -m "docs: publica resultados do benchmark unificado"
```

### Task 6: Verificação final do repositório

**Files:**
- Verify only.

**Interfaces:**
- Consumes: todos os artefatos anteriores.
- Produces: evidência final de correção, completude e árvore de trabalho limpa.

- [ ] **Step 1: Rodar a validação funcional final**

Run: `powershell -ExecutionPolicy Bypass -File scripts/run_benchmarks.ps1 -ValidarSomente`
Expected: PASS.

- [ ] **Step 2: Auditar o CSV final**

Run: importar `resultados/benchmark-raw.csv` no PowerShell e agrupar por versão/threads.
Expected: oito grupos, cinco linhas por grupo e nenhum campo vazio.

- [ ] **Step 3: Auditar requisitos textuais**

Run: verificar no readme compilação, execução, explicação do algoritmo, dataset real, tempo sequencial >= 10 s, OpenMP 1/2/4/8/16/32, CUDA, speedup, método e ambiente.
Expected: todos presentes.

- [ ] **Step 4: Conferir estado Git**

Run: `git status --short; git log --oneline -6`
Expected: árvore limpa e commits das Tasks 1-5 presentes (exceto Task 1, que não altera o repositório).
