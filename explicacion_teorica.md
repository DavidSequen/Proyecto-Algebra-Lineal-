# Explicación teórica — Compresor matricial de archivos .txt (formato .mtxc)

## 1. Fundamento teórico

**El archivo como datos numéricos:** todo archivo `.txt` es, a nivel de
computadora, una secuencia de bytes (enteros de 0 a 255). Esto permite
tratarlo como datos y aplicarle Álgebra Lineal.

**El anillo Z/256Z:** se trabaja en aritmética módulo 256 (enteros de 8
bits sin signo), exactamente como una computadora representa un byte.
En este anillo, un elemento tiene inverso multiplicativo si y solo si es
impar (coprimido con 256 = 2⁸). Para una matriz cuadrada A, esto se traduce
en: **A es invertible módulo 256 si y solo si det(A) es impar.** Todas
las matrices de este proyecto se construyen con determinante 1
(triangulares con unos en la diagonal, o permutaciones), por lo que
siempre son invertibles.

**Conceptos matemáticos aplicados:** vectores, matrices, multiplicación
matricial, matriz inversa, matrices de permutación, composición de
transformaciones lineales, y entropía de Shannon como métrica de
información.

## 2. Identificación de variables

| Variable | Símbolo | Unidad | Significado |
|---|---|---|---|
| Tamaño de bloque | N | bytes | Longitud de cada vector/bloque (N = 8) |
| Bloque de datos | x | valores 0–255 | Vector columna con N bytes consecutivos |
| Matriz de transformación | A | — | Matriz (o producto de matrices) N×N invertible mod 256 |
| Matriz inversa | A⁻¹ | — | Inversa de A; reconstruye el bloque original |
| Datos transformados | Y | valores 0–255 | Y = A·X (mod 256) |
| Entropía | H | bits/byte | Cota teórica de Shannon; mide qué tan comprimible es una representación |
| Longitud original | L | bytes | Tamaño del archivo antes del relleno |
| Tamaño transformado | T | bytes | Tamaño tras la transformación (T ≥ L; la matriz NO comprime) |
| Tamaño comprimido | C | bytes | Tamaño final del archivo `.mtxc` |

## 3. Formulación del modelo matemático

El archivo (longitud L) se rellena con ceros hasta ser múltiplo de N y se
organiza como matriz **X** (N filas × M columnas), cada columna un bloque.

**Matrices candidatas** (catálogo, todas invertibles mod 256):

- **D₁** (diferencia de primer orden): bidiagonal inferior, `D₁[i][i]=1`,
  `D₁[i][i-1]=-1 (=255 mod 256)`. Representa "diferencia entre valores
  consecutivos". Su inversa **D₁⁻¹** es la matriz triangular inferior de
  puros unos (suma acumulada).
- **R** (reversión de bloque): matriz de permutación que invierte el
  orden de los elementos. Es involutiva: **R·R = I**, por lo tanto es su
  propia inversa (R⁻¹ = R = Rᵀ).
- Combinaciones: D₁² (D₁ aplicada dos veces), D₁∘R, R∘D₁.

**Transformación general (una o varias matrices compuestas):**

```
Y = A_k · ... · A_2 · A_1 · X   (mod 256)
```

**Reconstrucción (inversas en orden contrario):**

```
X = A_1⁻¹ · A_2⁻¹ · ... · A_k⁻¹ · Y   (mod 256)
```

Se demuestra (y se verifica en el código) que `A_total⁻¹ · A_total = I`.

**Selección objetiva:** para cada archivo, se calcula la **entropía de
Shannon** de cada candidato:

```
H = - Σ p(v) · log2(p(v))
```

donde `p(v)` es la frecuencia relativa de cada valor de byte. Un H menor
significa que Huffman podrá comprimir mejor (H es la cota teórica inferior
del número de bits promedio por símbolo, según el teorema de codificación
de fuentes de Shannon). Se elige el candidato con **menor H** — una
métrica matemática, no una apreciación subjetiva.

## 4. Aplicación de Álgebra Lineal y desarrollo de cálculos

1. Construir X (bloques como columnas).
2. Probar cada pipeline candidato (identidad, D₁, D₁², R, D₁∘R, R∘D₁):
   calcular Y = A·X y su entropía H(Y).
3. Elegir el pipeline con menor H (métrica objetiva).
4. Codificar Y (o X, si ganó la identidad) con Huffman: los valores más
   frecuentes reciben códigos más cortos. **Esta etapa NO es Álgebra
   Lineal**; es la que realmente reduce el tamaño en bytes.
5. Si el resultado comprimido no es más pequeño que el original, se
   almacena el archivo tal cual (modo "almacenado"), garantizando que el
   `.mtxc` nunca crezca de forma significativa.
6. Para descomprimir: decodificar Huffman, y si se usó una transformación,
   calcular X = A₁⁻¹·A₂⁻¹·...·Aₖ⁻¹·Y para deshacerla exactamente.
7. Verificar byte a byte que el archivo reconstruido sea idéntico al
   original.

**Ejemplo a mano (N = 4):** vector `x = [65, 65, 65, 66]`

```
D1 = [[1,0,0,0],[255,1,0,0],[0,255,1,0],[0,0,255,1]]
Y = D1·x = [65, 0, 0, 1]
X = D1⁻¹·Y = [65, 65, 65, 66]   ← idéntico al original
```

## 5. Resultado experimental honesto (hallazgo matemático real)

Se probó el sistema con datos reales, incluyendo un archivo tabular con
campos de ancho fijo y relleno de espacios, pensado específicamente para
favorecer a D₁. **Resultado: en todos los archivos de texto natural
probados, ganó la Identidad (no transformar).**

**Por qué ocurre esto, y por qué es un resultado válido, no una falla:**
las matrices D₁ y sus combinaciones funcionan bien cuando existe
*correlación numérica local* entre bytes consecutivos (por ejemplo, una
señal suave, o texto generado con secuencias de códigos incrementales).
El texto natural en español no tiene esa propiedad: los códigos ASCII de
letras consecutivas no guardan relación numérica cercana entre sí (por
ejemplo, "a" y " " distan 65 unidades), así que diferenciar **dispersa**
la entropía en vez de concentrarla. Esto se comprobó también en sentido
contrario: con un archivo de prueba diseñado con codificación secuencial
(bytes que incrementan de 1 en 1 dentro de cada bloque), **D₁ sí ganó**,
bajando la entropía de 3.000 a 0.544 bits/byte, con una reducción de
tamaño del 87.4% y reconstrucción exacta.

Esto demuestra que:
1. El mecanismo de selección de matrices es **funcional y objetivo**, no
   decorativo: se activa cuando corresponde y se puede probar que
   funciona en ambos sentidos.
2. La elección "no transformar" también es una decisión matemática
   legítima, sustentada en la métrica de entropía, no una falla del
   programa.
3. La compresión real siempre la realiza Huffman; las matrices deciden
   *qué representación* le conviene más a Huffman.

## 6. Formato del archivo `.mtxc`

```
MAGIC (4B, "MTXC") + VERSIÓN (1B) + MODO (1B)

MODO 2 = archivo vacío            → sin cuerpo adicional
MODO 0 = almacenado sin comprimir → bytes originales tal cual
MODO 1 = comprimido:
    N (1B) + longitud_original (4B) + total_valores_transformados (4B)
    + núm_matrices (1B) + ids_matrices (núm_matrices B)
    + relleno_bits (1B) + (núm_símbolos - 1) (1B)
    + [símbolo(1B) + frecuencia(4B)] × núm_símbolos   ← tabla COMPACTA
    + flujo de bits Huffman
```

La tabla de frecuencias solo guarda los símbolos que realmente aparecen
en el archivo (nunca las 256 entradas fijas), evitando almacenar
información redundante innecesaria — importante para archivos pequeños
o con pocos símbolos distintos.

## 7. Conclusiones preliminares

1. Una matriz invertible en Z/256Z permite transformar datos de forma
   completamente reversible, con determinante impar garantizando la
   invertibilidad.
2. La composición de matrices (A₃A₂A₁) es igualmente invertible, y su
   inversa se obtiene invirtiendo cada factor en orden contrario —
   propiedad demostrada y verificada automáticamente en el código.
3. La entropía de Shannon es una métrica objetiva y matemáticamente
   fundamentada para decidir si una transformación conviene, sin
   necesidad de juicios subjetivos.
4. La eficacia de una transformación depende de la estructura estadística
   de los datos: útil para datos con correlación numérica local, neutra
   o contraproducente para texto natural — un resultado verificable, no
   una limitación oculta.
5. La compresión real (reducción de bytes) la logra la codificación de
   Huffman; el papel del Álgebra Lineal es preparar y, cuando conviene,
   optimizar la representación que Huffman va a codificar.
