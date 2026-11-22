# Benchmark unificado do K-means

## Intenção

Produzir tempos comparáveis para as versões sequencial, OpenMP e CUDA do
K-means, todos obtidos na mesma máquina, com o mesmo dataset, parâmetros e
região cronometrada. Os resultados serão usados no trabalho de Computação
Paralela e deverão permitir explicar tanto a correção quanto o speedup.

## Objetivos

- Compilar as três versões com otimização equivalente e ferramentas locais.
- Usar `data/covtype.bin`, `K = 32` e `max_iter = 300` em todos os testes.
- Cronometrar apenas o laço iterativo do K-means; leitura, inicialização e SSE
  ficam fora, assim como a transferência inicial para a GPU.
- Validar iterações, SSE e tamanhos dos clusters antes de aceitar os tempos.
- Medir sequencial, CUDA e OpenMP com 1, 2, 4, 8, 16 e 32 threads.
- Repetir cada configuração e publicar a mediana, calculando todos os speedups
  contra a nova mediana sequencial.
- Atualizar os cabeçalhos dos três códigos e o `readme.txt` com método,
  ambiente, resultados e comandos reproduzíveis.

## Não objetivos e antiobjetivos

- Não modificar o algoritmo ou reduzir o trabalho apenas para produzir um
  speedup maior.
- Não reutilizar tempos antigos nem combinar medições de máquinas diferentes.
- Não incluir leitura de arquivo ou cálculo de SSE no tempo principal.
- Não afirmar escalabilidade para 32 threads físicas: a CPU atual tem 8
  núcleos e 16 threads lógicas, portanto 32 threads OpenMP representam
  sobrescrição e serão identificadas dessa forma.

## Restrições

- Máquina de medição: AMD Ryzen 7 5700X3D, 8 núcleos/16 threads lógicas,
  NVIDIA GeForce RTX 4060 Ti e Windows 11 Pro.
- O requisito acadêmico exige os resultados OpenMP para 1, 2, 4, 8, 16 e 32
  threads e uma versão CUDA.
- O CUDA Toolkit precisa ser instalado antes da compilação CUDA.
- O dataset real sequencial deve continuar demandando pelo menos 10 segundos.

## Abordagens consideradas

1. **Benchmark parcial imediato.** Medir sequencial/OpenMP agora e manter o
   tempo CUDA antigo. É mais rápido, mas mantém a comparação inválida entre
   máquinas; foi descartado.
2. **Benchmark integral na máquina atual.** Instalar o CUDA Toolkit, compilar e
   medir todas as versões localmente com repetições e mediana. Foi escolhido
   porque elimina a principal fonte de inconsistência e produz uma tabela
   defensável.

## Método escolhido

Primeiro serão compilados e executados testes de correção. Divergências serão
investigadas antes das medições. Cada configuração terá uma execução de
aquecimento e múltiplas execuções medidas; a quantidade será fixada no script
de benchmark e documentada no readme. Durante a série, serão preservados os
mesmos binários, dataset, parâmetros e região cronometrada. A mediana será
usada para reduzir o efeito de processos do sistema e aquecimento do hardware.

Os resultados brutos deverão ser preservados em um arquivo de saída gerado
pelo benchmark, enquanto a tabela consolidada será inserida nos comentários
iniciais de `kmeans_seq.c`, `kmeans_omp.c`, `kmeans_cuda.cu` e no
`readme.txt`.

## Critérios de aceitação

- As três versões compilam na máquina documentada.
- Todas convergem de forma funcionalmente compatível e produzem SSE e
  distribuições de clusters justificadamente equivalentes.
- A tabela final contém sequencial, OpenMP 1/2/4/8/16/32 e CUDA, sem campos em
  branco e com speedups recalculados.
- O procedimento pode ser repetido pelos comandos do `readme.txt`.
- A versão sequencial medida atende ao mínimo de 10 segundos.
