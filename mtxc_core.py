"""
mtxc_core.py — Motor común del proyecto .mtxc

Contiene TODO lo que comparten los tres programas:
    - compresor_texto.py        (archivos de texto / cualquier archivo genérico)
    - compresor_imagenes.py     (imágenes, con transformación matricial 2D real)
    - compresor_multiarchivo.py (empaquetado de varios archivos en un .mtxa)

Nada en este archivo sabe de GUI ni de tipos de archivo específicos: son
las piezas matemáticas y de codificación, reutilizables.

═══════════════════════════════════════════════════════════════════════
QUÉ VIVE AQUÍ
═══════════════════════════════════════════════════════════════════════
1. Aritmética Z/256Z y matrices candidatas (1D, por bloques de N bytes).
2. Entropía de Shannon (métrica objetiva de selección).
3. Codificación de Huffman (compresión real, sin pérdida).
4. Funciones de bytes <-> registro comprimido, reutilizables por
   cualquier programa que tenga "unos bytes que comprimir" —
   sin importar si vienen de un .txt, de los píxeles de una imagen, o
   de varios archivos concatenados.
"""

import heapq
import struct
import numpy as np

MOD = 256
BLOCK_SIZE = 8
PAD_BYTE = 0
MAGIC = b"MTXC"
VERSION = 3

# ═══════════════════════════════════════════════════════════════════
# 1) MATRICES CANDIDATAS (Z/256Z) — transformación 1D por bloques
# ═══════════════════════════════════════════════════════════════════
def construir_identidad(n):
    return np.eye(n, dtype=np.uint8)


def construir_diferencia1(n):
    """D1: y[0]=x[0], y[i]=x[i]-x[i-1]. Bidiagonal, det=1 (invertible)."""
    A = np.eye(n, dtype=np.uint8)
    for i in range(1, n):
        A[i][i - 1] = MOD - 1  # -1 mod 256
    return A


def construir_diferencia1_inversa(n):
    """Inversa de D1: suma acumulada (triangular inferior de unos)."""
    return np.tril(np.ones((n, n), dtype=np.uint8))


def construir_reversion(n):
    """R: invierte el orden dentro del bloque. Involutiva (R@R=I)."""
    A = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        A[i][n - 1 - i] = 1
    return A


def catalogo_matrices(n):
    D1 = construir_diferencia1(n)
    D1_inv = construir_diferencia1_inversa(n)
    R = construir_reversion(n)
    return {
        1: ("D1 (diferencia 1er orden)", D1, D1_inv),
        3: ("R (reversión de bloque)", R, R),
    }


PIPELINES_CANDIDATOS = [[], [1], [1, 1], [3], [1, 3], [3, 1]]


def nombre_pipeline(ids, catalogo):
    if not ids:
        return "Identidad (sin transformar)"
    return " -> ".join(catalogo[i][0] for i in ids)


def aplicar_pipeline(X, ids, catalogo):
    Y = X
    for i in ids:
        _, A, _ = catalogo[i]
        Y = A @ Y
    return Y


def deshacer_pipeline(Y, ids, catalogo):
    X = Y
    for i in reversed(ids):
        _, _, A_inv = catalogo[i]
        X = A_inv @ X
    return X


def verificar_inversa_pipeline(ids, n):
    catalogo = catalogo_matrices(n)
    identidad = np.eye(n, dtype=np.uint8)
    prueba = np.eye(n, dtype=np.uint8)
    prueba = aplicar_pipeline(prueba, ids, catalogo)
    prueba = deshacer_pipeline(prueba, ids, catalogo)
    return np.array_equal(prueba, identidad)


# ═══════════════════════════════════════════════════════════════════
# 2) MATRICES 2D PARA BLOQUES DE IMAGEN (B×B), Y = D·X·Dᵀ (mod 256)
# ═══════════════════════════════════════════════════════════════════
def construir_diferencia2d_inversa(b):
    """Devuelve (D, D_inv) de tamaño B×B para transformar bloques 2D
    de imagen. Se aplica como Y = D·X·Dᵀ (filas Y columnas a la vez,
    multiplicación matricial real de matrices B×B con el bloque)."""
    D = construir_diferencia1(b)
    D_inv = construir_diferencia1_inversa(b)
    return D, D_inv


def transformar_bloque2d(X, D):
    """Y = D · X · Dᵀ (mod 256). X es un bloque B×B."""
    return (D @ X) @ D.T


def destransformar_bloque2d(Y, D_inv):
    """X = D⁻¹ · Y · (D⁻¹)ᵀ (mod 256)."""
    return (D_inv @ Y) @ D_inv.T


# ═══════════════════════════════════════════════════════════════════
# 3) BYTES <-> BLOQUES 1D
# ═══════════════════════════════════════════════════════════════════
def bytes_a_matriz_bloques(datos: bytes, n: int):
    longitud_original = len(datos)
    if longitud_original == 0:
        return np.zeros((n, 0), dtype=np.uint8), 0
    num_bloques = -(-longitud_original // n)
    total_relleno = num_bloques * n
    arreglo = np.frombuffer(datos, dtype=np.uint8)
    if total_relleno > longitud_original:
        relleno = np.full(total_relleno - longitud_original, PAD_BYTE, dtype=np.uint8)
        arreglo = np.concatenate([arreglo, relleno])
    return arreglo.reshape(num_bloques, n).T, longitud_original


def matriz_bloques_a_bytes(X: np.ndarray, longitud_original: int) -> bytes:
    if longitud_original == 0:
        return b""
    arreglo = X.T.reshape(-1)[:longitud_original]
    return arreglo.astype(np.uint8).tobytes()


# ═══════════════════════════════════════════════════════════════════
# 4) ENTROPÍA DE SHANNON — métrica objetiva
# ═══════════════════════════════════════════════════════════════════
def entropia_bits_por_byte(valores: np.ndarray) -> float:
    if len(valores) == 0:
        return 0.0
    frecuencias = np.bincount(valores, minlength=256).astype(np.float64)
    frecuencias = frecuencias[frecuencias > 0]
    probabilidades = frecuencias / frecuencias.sum()
    return float(-(probabilidades * np.log2(probabilidades)).sum())


def seleccionar_mejor_transformacion(X: np.ndarray, n: int):
    """Prueba cada pipeline candidato 1D, mide entropía, regresa el mejor."""
    catalogo = catalogo_matrices(n)
    reporte = []
    mejor = None
    for ids in PIPELINES_CANDIDATOS:
        Y = aplicar_pipeline(X, ids, catalogo)
        h = entropia_bits_por_byte(Y.T.reshape(-1))
        reporte.append((ids, nombre_pipeline(ids, catalogo), h))
        if mejor is None or h < mejor[2]:
            mejor = (ids, Y, h)
    return mejor[0], mejor[1], reporte


# ═══════════════════════════════════════════════════════════════════
# 5) CODIFICACIÓN DE HUFFMAN (no es Álgebra Lineal)
# ═══════════════════════════════════════════════════════════════════
def construir_codigos_huffman(frecuencias: dict) -> dict:
    heap = []
    contador = 0
    for simbolo, freq in frecuencias.items():
        heapq.heappush(heap, (int(freq), contador, simbolo, None, None))
        contador += 1
    if len(heap) == 0:
        return {}
    if len(heap) == 1:
        _, _, simbolo, _, _ = heap[0]
        return {simbolo: "0"}
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        heapq.heappush(heap, (a[0] + b[0], contador, None, a, b))
        contador += 1
    codigos = {}
    pila = [(heap[0], "")]
    while pila:
        nodo, prefijo = pila.pop()
        _, _, simbolo, izq, der = nodo
        if simbolo is not None:
            codigos[simbolo] = prefijo or "0"
        else:
            pila.append((izq, prefijo + "0"))
            pila.append((der, prefijo + "1"))
    return codigos


def huffman_codificar(valores: np.ndarray, codigos: dict):
    tabla = [codigos.get(b, "") for b in range(256)]
    bitstring = "".join(tabla[b] for b in valores)
    relleno = (8 - len(bitstring) % 8) % 8
    bitstring += "0" * relleno
    cuerpo = int(bitstring, 2).to_bytes(len(bitstring) // 8, "big") if bitstring else b""
    return cuerpo, relleno


def huffman_decodificar(cuerpo: bytes, relleno: int, frecuencias: dict, total_valores: int) -> np.ndarray:
    codigos = construir_codigos_huffman(frecuencias)

    class Nodo:
        __slots__ = ("izq", "der", "simbolo")

        def __init__(self):
            self.izq = None
            self.der = None
            self.simbolo = None

    raiz = Nodo()
    for simbolo, codigo in codigos.items():
        nodo = raiz
        for bit in codigo:
            if bit == "0":
                nodo.izq = nodo.izq or Nodo()
                nodo = nodo.izq
            else:
                nodo.der = nodo.der or Nodo()
                nodo = nodo.der
        nodo.simbolo = simbolo

    total_bits = len(cuerpo) * 8
    bits = bin(int.from_bytes(cuerpo, "big"))[2:].zfill(total_bits) if cuerpo else ""
    bits_utiles = len(bits) - relleno

    salida = np.empty(total_valores, dtype=np.uint8)
    nodo = raiz
    idx_salida = 0
    for i in range(bits_utiles):
        nodo = nodo.izq if bits[i] == "0" else nodo.der
        if nodo.simbolo is not None:
            salida[idx_salida] = nodo.simbolo
            idx_salida += 1
            nodo = raiz
            if idx_salida == total_valores:
                break
    return salida


# ═══════════════════════════════════════════════════════════════════
# 6) BYTES -> REGISTRO COMPRIMIDO (reutilizable por cualquier programa)
#
#    Un "registro" es la representación comprimida de UN blob de bytes,
#    SIN encabezado de archivo (eso lo agrega cada programa). Formato:
#
#    MODO(1B) +
#      si MODO=2: nada (blob vacío)
#      si MODO=0: los bytes originales tal cual (no convino comprimir)
#      si MODO=1: N(1B) + long_original(4B) + total_valores(4B)
#                 + num_matrices(1B) + ids(num_matrices B)
#                 + relleno_bits(1B) + (num_simbolos-1)(1B)
#                 + [simbolo(1B)+frecuencia(4B)] * num_simbolos
#                 + flujo Huffman
# ═══════════════════════════════════════════════════════════════════
def comprimir_bytes(datos: bytes, n: int = BLOCK_SIZE) -> dict:
    """Comprime un blob de bytes genérico. Regresa un dict con el
    'registro' (bytes listos para guardar) y metadatos para reportar."""
    tam_original = len(datos)
    if tam_original == 0:
        return {"registro": struct.pack("<B", 2), "tam_original": 0, "tam_transformado": 0,
                "modo": "vacio", "nombre_pipeline": "N/A", "reporte_entropia": []}

    X, longitud_original = bytes_a_matriz_bloques(datos, n)
    ids_elegidos, Y, reporte_entropia = seleccionar_mejor_transformacion(X, n)
    catalogo = catalogo_matrices(n)

    valores_finales = Y.T.reshape(-1)
    tam_transformado = len(valores_finales)

    frecuencias_arr = np.bincount(valores_finales, minlength=256)
    frecuencias = {int(s): int(c) for s, c in enumerate(frecuencias_arr) if c > 0}
    codigos = construir_codigos_huffman(frecuencias)
    cuerpo, relleno = huffman_codificar(valores_finales, codigos)

    tabla_frecuencias = b"".join(struct.pack("<BI", s, c) for s, c in frecuencias.items())
    cuerpo_modo1 = (
        struct.pack("<BIIB", n, longitud_original, tam_transformado, len(ids_elegidos))
        + bytes(ids_elegidos)
        + struct.pack("<BB", relleno, len(frecuencias) - 1)
        + tabla_frecuencias
        + cuerpo
    )

    if 1 + len(cuerpo_modo1) < 1 + tam_original:
        registro = struct.pack("<B", 1) + cuerpo_modo1
        modo = "comprimido"
        pipeline_usado = ids_elegidos
    else:
        registro = struct.pack("<B", 0) + datos
        modo = "almacenado"
        pipeline_usado = []

    return {
        "registro": registro,
        "tam_original": tam_original,
        "tam_transformado": tam_transformado,
        "tam_comprimido": len(registro),
        "modo": modo,
        "nombre_pipeline": nombre_pipeline(pipeline_usado, catalogo),
        "reporte_entropia": [(nombre_pipeline(ids, catalogo), h) for ids, _, h in reporte_entropia],
    }


def descomprimir_bytes(registro: bytes) -> bytes:
    """Inverso exacto de comprimir_bytes: registro -> bytes originales."""
    modo = registro[0]
    cuerpo = registro[1:]
    if modo == 2:
        return b""
    if modo == 0:
        return cuerpo

    n, longitud_original, total_valores, num_matrices = struct.unpack("<BIIB", cuerpo[:10])
    cursor = 10
    ids = list(cuerpo[cursor:cursor + num_matrices])
    cursor += num_matrices
    relleno, num_simbolos_menos_1 = struct.unpack("<BB", cuerpo[cursor:cursor + 2])
    cursor += 2
    num_simbolos = num_simbolos_menos_1 + 1
    frecuencias = {}
    for _ in range(num_simbolos):
        simbolo, cuenta = struct.unpack("<BI", cuerpo[cursor:cursor + 5])
        frecuencias[simbolo] = cuenta
        cursor += 5
    cuerpo_huffman = cuerpo[cursor:]

    valores_finales = huffman_decodificar(cuerpo_huffman, relleno, frecuencias, total_valores)
    num_bloques = total_valores // n if n else 0
    Y = valores_finales.reshape(num_bloques, n).T

    catalogo = catalogo_matrices(n)
    X = deshacer_pipeline(Y, ids, catalogo)
    return matriz_bloques_a_bytes(X, longitud_original)


def archivos_son_identicos(ruta_a: str, ruta_b: str) -> bool:
    with open(ruta_a, "rb") as fa, open(ruta_b, "rb") as fb:
        return fa.read() == fb.read()
