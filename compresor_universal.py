"""
compresor_universal.py — Compresor matricial para CUALQUIER archivo (.mtxu)

Este programa acepta cualquier archivo: documentos, imágenes, binarios,
lo que sea. Internamente decide, según el tipo de dato, cuál motor usar:

    - Si el archivo es un BMP de 24 bits sin comprimir (BI_RGB): usa la
      transformación matricial 2D real, Y = D·X·Dᵀ (mod 256), sobre
      bloques de 8×8 píxeles de cada canal de color. Ver el bloque de
      comentarios más abajo para el porqué matemático.
    - Para cualquier otro archivo (documentos, PDFs, PNG/JPG ya
      comprimidos, ejecutables, lo que sea): usa el motor genérico de
      mtxc_core (matrices 1D por bloques de bytes + selección objetiva
      por entropía + Huffman) — el mismo usado por compresor_texto.py.

═══════════════════════════════════════════════════════════════════════
POR QUÉ LAS IMÁGENES BMP SE TRATAN DISTINTO A TODO LO DEMÁS
═══════════════════════════════════════════════════════════════════════
Un archivo de texto (o binario genérico) es, en esencia, una lista de
valores (1D). Una imagen sin comprimir es, por naturaleza, una MATRIZ
de dos dimensiones: filas y columnas de píxeles. Ahí la transformación
matricial deja de ser una elección arbitraria y pasa a ser la
representación NATURAL del dato:

    Y = D · X · Dᵀ   (mod 256)

donde X es un bloque de 8×8 píxeles de UN canal de color (B, G o R),
D es la misma matriz de diferencias bidiagonal del resto del proyecto,
y la multiplicación se hace POR AMBOS LADOS: D·X aplica la diferencia
a las FILAS del bloque, y luego ·Dᵀ aplica la diferencia a las
COLUMNAS. Esto es exactamente lo que hacen los codecs de imagen sin
pérdida reales (por ejemplo, la predicción espacial de PNG o de FFV1)
para preparar los píxeles antes de comprimirlos: los píxeles vecinos
en una foto normal casi siempre tienen valores parecidos (a diferencia
de las letras de un texto), así que diferenciar SÍ concentra la
información — es la situación exactamente opuesta a la que
encontramos con texto.

Reconstrucción exacta:

    X = D⁻¹ · Y · (D⁻¹)ᵀ   (mod 256)

═══════════════════════════════════════════════════════════════════════
POR QUÉ SOLO BMP SIN COMPRIMIR RECIBE EL TRATO 2D
═══════════════════════════════════════════════════════════════════════
El requisito de este proyecto es que la reconstrucción sea idéntica
BYTE A BYTE al archivo original — no solo que la imagen se vea igual.

PNG y JPG ya vienen comprimidos por su propio formato. Para editar sus
píxeles habría que decodificarlos y luego volver a codificarlos, y
ningún codificador (ni el nuestro) puede garantizar reproducir los
bytes EXACTOS del archivo original — solo una imagen visualmente
idéntica, que no es lo que este proyecto exige. Por eso esos formatos
(y cualquier documento u otro archivo) usan el motor genérico, que
sigue garantizando compresión sin pérdida — simplemente sin el
tratamiento 2D especial, y así se le informa al usuario.
"""

import struct
import numpy as np
import mtxc_core as core

MAGIC = b"MTXU"
VERSION = 1
BLOCK2D = 8  # tamaño de bloque 2D (como los bloques 8x8 de JPEG)
EXT_COMPRIMIDO = ".mtxu"


# ═══════════════════════════════════════════════════════════════════
# 1) LECTURA / ESCRITURA DE BMP (sin comprimir, 24 bits) — a mano,
#    sin librerías de imagen, para poder garantizar bytes exactos.
# ═══════════════════════════════════════════════════════════════════
def leer_bmp_24bit(datos: bytes):
    """Intenta interpretar 'datos' como un BMP de 24 bits, BI_RGB (sin
    comprimir). Devuelve un dict con toda la información necesaria
    para reconstruirlo exacto, o None si no es compatible."""
    if len(datos) < 54 or datos[0:2] != b"BM":
        return None

    offset_pixeles = struct.unpack("<I", datos[10:14])[0]
    tam_dib = struct.unpack("<I", datos[14:18])[0]
    if tam_dib != 40:  # solo BITMAPINFOHEADER estándar
        return None

    ancho = struct.unpack("<i", datos[18:22])[0]
    alto_crudo = struct.unpack("<i", datos[22:26])[0]
    bpp = struct.unpack("<H", datos[28:30])[0]
    compresion = struct.unpack("<I", datos[30:34])[0]

    if bpp != 24 or compresion != 0:
        return None  # no soportado -> se usará el modo genérico

    alto = abs(alto_crudo)
    fila_sin_relleno = ancho * 3
    stride = ((fila_sin_relleno + 3) // 4) * 4
    relleno = stride - fila_sin_relleno

    total_pixeles = stride * alto
    if offset_pixeles + total_pixeles > len(datos):
        return None

    encabezado = datos[:offset_pixeles]        # guardado verbatim
    cuerpo_extra = datos[offset_pixeles + total_pixeles:]  # datos tras los píxeles (si los hay)
    bloque_pixeles = datos[offset_pixeles: offset_pixeles + total_pixeles]

    filas = [bloque_pixeles[f * stride: f * stride + stride] for f in range(alto)]
    canales_bgr = [np.zeros((alto, ancho), dtype=np.uint8) for _ in range(3)]
    relleno_por_fila = []
    for f, fila in enumerate(filas):
        pix = np.frombuffer(fila[:fila_sin_relleno], dtype=np.uint8).reshape(ancho, 3)
        canales_bgr[0][f, :] = pix[:, 0]
        canales_bgr[1][f, :] = pix[:, 1]
        canales_bgr[2][f, :] = pix[:, 2]
        relleno_por_fila.append(fila[fila_sin_relleno:])

    return {
        "encabezado": encabezado,
        "cuerpo_extra": cuerpo_extra,
        "ancho": ancho, "alto": alto,
        "relleno_por_fila": b"".join(relleno_por_fila),
        "tam_relleno_fila": relleno,
        "canales_bgr": canales_bgr,
    }


def reconstruir_bmp_24bit(info: dict) -> bytes:
    ancho, alto = info["ancho"], info["alto"]
    tam_relleno_fila = info["tam_relleno_fila"]
    relleno_bytes = info["relleno_por_fila"]
    canales = info["canales_bgr"]

    filas = []
    for f in range(alto):
        pix = np.stack([canales[0][f], canales[1][f], canales[2][f]], axis=1).astype(np.uint8)
        fila_bytes = pix.tobytes()
        if tam_relleno_fila:
            fila_bytes += relleno_bytes[f * tam_relleno_fila:(f + 1) * tam_relleno_fila]
        filas.append(fila_bytes)

    return info["encabezado"] + b"".join(filas) + info["cuerpo_extra"]


# ═══════════════════════════════════════════════════════════════════
# 2) TRANSFORMACIÓN 2D POR BLOQUES DE UN CANAL (matriz H×W completa)
# ═══════════════════════════════════════════════════════════════════
def canal_a_bloques2d(canal: np.ndarray, b: int):
    """Rellena el canal para que sea múltiplo de b en ambas
    dimensiones y lo corta en bloques b×b. Devuelve la lista de
    bloques y las dimensiones originales."""
    alto, ancho = canal.shape
    alto_r = -(-alto // b) * b
    ancho_r = -(-ancho // b) * b
    relleno = np.zeros((alto_r, ancho_r), dtype=np.uint8)
    relleno[:alto, :ancho] = canal
    bloques = []
    for i in range(0, alto_r, b):
        for j in range(0, ancho_r, b):
            bloques.append(relleno[i:i + b, j:j + b])
    return bloques, alto, ancho, alto_r, ancho_r


def bloques2d_a_canal(bloques, alto, ancho, alto_r, ancho_r, b):
    relleno = np.zeros((alto_r, ancho_r), dtype=np.uint8)
    idx = 0
    for i in range(0, alto_r, b):
        for j in range(0, ancho_r, b):
            relleno[i:i + b, j:j + b] = bloques[idx]
            idx += 1
    return relleno[:alto, :ancho]


def transformar_canal(canal: np.ndarray, D, usa_transform: bool):
    bloques, alto, ancho, alto_r, ancho_r = canal_a_bloques2d(canal, BLOCK2D)
    if usa_transform:
        bloques = [core.transformar_bloque2d(bq, D) for bq in bloques]
    return bloques, alto, ancho, alto_r, ancho_r


def destransformar_canal(bloques, alto, ancho, alto_r, ancho_r, D_inv, usa_transform: bool):
    if usa_transform:
        bloques = [core.destransformar_bloque2d(bq, D_inv) for bq in bloques]
    return bloques2d_a_canal(bloques, alto, ancho, alto_r, ancho_r, BLOCK2D)


# ═══════════════════════════════════════════════════════════════════
# 3) COMPRIMIR / DESCOMPRIMIR IMAGEN
# ═══════════════════════════════════════════════════════════════════
def comprimir_archivo(ruta_entrada: str, ruta_salida: str):
    with open(ruta_entrada, "rb") as f:
        datos = f.read()

    info_bmp = leer_bmp_24bit(datos)

    if info_bmp is None:
        # No es un BMP de 24 bits sin comprimir -> modo genérico honesto
        resultado = core.comprimir_bytes(datos)
        with open(ruta_salida, "wb") as f:
            f.write(MAGIC + struct.pack("<BB", VERSION, 0))  # modo 0 = genérico
            f.write(resultado["registro"])
        return {
            "modo_imagen": "genérico (no es BMP de 24 bits sin comprimir)",
            "tam_original": len(datos),
            "tam_comprimido": 6 + len(resultado["registro"]),
            "usa_transform_2d": False,
        }

    D, D_inv = core.construir_diferencia2d_inversa(BLOCK2D)

    # probar CON y SIN transformación 2D, medir entropía, elegir la mejor
    valores_por_opcion = {}
    bloques_por_canal = {}
    for usa in (False, True):
        todos_los_valores = []
        for canal in info_bmp["canales_bgr"]:
            bloques, alto, ancho, alto_r, ancho_r = transformar_canal(canal, D, usa)
            bloques_por_canal.setdefault(usa, []).append((bloques, alto, ancho, alto_r, ancho_r))
            for bq in bloques:
                todos_los_valores.append(bq.reshape(-1))
        valores_por_opcion[usa] = np.concatenate(todos_los_valores)

    h_sin = core.entropia_bits_por_byte(valores_por_opcion[False])
    h_con = core.entropia_bits_por_byte(valores_por_opcion[True])
    usa_transform = h_con < h_sin
    valores_finales = valores_por_opcion[usa_transform]

    frecuencias_arr = np.bincount(valores_finales, minlength=256)
    frecuencias = {int(s): int(c) for s, c in enumerate(frecuencias_arr) if c > 0}
    codigos = core.construir_codigos_huffman(frecuencias)
    cuerpo_huff, relleno_bits = core.huffman_codificar(valores_finales, codigos)

    tabla_frecuencias = b"".join(struct.pack("<BI", s, c) for s, c in frecuencias.items())

    cuerpo = (
        struct.pack("<I", len(info_bmp["encabezado"])) + info_bmp["encabezado"]
        + struct.pack("<I", len(info_bmp["cuerpo_extra"])) + info_bmp["cuerpo_extra"]
        + struct.pack("<HHB", info_bmp["ancho"], info_bmp["alto"], info_bmp["tam_relleno_fila"])
        + struct.pack("<I", len(info_bmp["relleno_por_fila"])) + info_bmp["relleno_por_fila"]
        + struct.pack("<B", int(usa_transform))
        + struct.pack("<IB", len(valores_finales), relleno_bits)
        + struct.pack("<B", len(frecuencias) - 1) + tabla_frecuencias
        + cuerpo_huff
    )

    registro_completo = MAGIC + struct.pack("<BB", VERSION, 1) + cuerpo  # modo 1 = imagen 2D

    tam_original = len(datos)
    if len(registro_completo) >= tam_original:
        # ni siquiera Huffman ayudó (imagen muy poco redundante) -> almacenar tal cual
        with open(ruta_salida, "wb") as f:
            f.write(MAGIC + struct.pack("<BB", VERSION, 2) + datos)  # modo 2 = almacenado
        return {"modo_imagen": "almacenado (comprimir no ayudaba)", "tam_original": tam_original,
                "tam_comprimido": 6 + tam_original, "usa_transform_2d": False}

    with open(ruta_salida, "wb") as f:
        f.write(registro_completo)

    return {
        "modo_imagen": "matricial 2D" if usa_transform else "sin transformar (Huffman directo)",
        "tam_original": tam_original,
        "tam_comprimido": len(registro_completo),
        "usa_transform_2d": usa_transform,
        "entropia_sin_transformar": h_sin,
        "entropia_con_transformar": h_con,
    }


def descomprimir_archivo(ruta_comprimida: str, ruta_salida: str):
    with open(ruta_comprimida, "rb") as f:
        contenido = f.read()

    if contenido[:4] != MAGIC:
        raise ValueError("El archivo no tiene la firma .mtxu esperada")
    version, modo = struct.unpack("<BB", contenido[4:6])
    cuerpo = contenido[6:]

    if modo == 0:
        datos = core.descomprimir_bytes(cuerpo)
    elif modo == 2:
        datos = cuerpo
    else:
        cursor = 0
        (n1,) = struct.unpack_from("<I", cuerpo, cursor); cursor += 4
        encabezado = cuerpo[cursor:cursor + n1]; cursor += n1
        (n2,) = struct.unpack_from("<I", cuerpo, cursor); cursor += 4
        cuerpo_extra = cuerpo[cursor:cursor + n2]; cursor += n2
        ancho, alto, tam_relleno_fila = struct.unpack_from("<HHB", cuerpo, cursor); cursor += 5
        (n3,) = struct.unpack_from("<I", cuerpo, cursor); cursor += 4
        relleno_por_fila = cuerpo[cursor:cursor + n3]; cursor += n3
        (usa_transform,) = struct.unpack_from("<B", cuerpo, cursor); cursor += 1
        total_valores, relleno_bits = struct.unpack_from("<IB", cuerpo, cursor); cursor += 5
        (num_simb_menos1,) = struct.unpack_from("<B", cuerpo, cursor); cursor += 1
        num_simbolos = num_simb_menos1 + 1
        frecuencias = {}
        for _ in range(num_simbolos):
            s, c = struct.unpack_from("<BI", cuerpo, cursor); cursor += 5
            frecuencias[s] = c
        cuerpo_huff = cuerpo[cursor:]

        valores = core.huffman_decodificar(cuerpo_huff, relleno_bits, frecuencias, total_valores)

        b = BLOCK2D
        alto_r = -(-alto // b) * b
        ancho_r = -(-ancho // b) * b
        bloques_por_dim = (alto_r // b) * (ancho_r // b)
        D, D_inv = core.construir_diferencia2d_inversa(b)

        canales = []
        pos = 0
        for _c in range(3):
            bloques = []
            for _k in range(bloques_por_dim):
                bloque = valores[pos:pos + b * b].reshape(b, b)
                pos += b * b
                bloques.append(bloque)
            canal = destransformar_canal(bloques, alto, ancho, alto_r, ancho_r, D_inv, bool(usa_transform))
            canales.append(canal)

        info = {
            "encabezado": encabezado, "cuerpo_extra": cuerpo_extra,
            "ancho": ancho, "alto": alto,
            "relleno_por_fila": relleno_por_fila, "tam_relleno_fila": tam_relleno_fila,
            "canales_bgr": canales,
        }
        datos = reconstruir_bmp_24bit(info)

    with open(ruta_salida, "wb") as f:
        f.write(datos)
    return datos


# ═══════════════════════════════════════════════════════════════════
# 4) GUI
# ═══════════════════════════════════════════════════════════════════
def iniciar_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox

    ventana = tk.Tk()
    ventana.title("Compresor matricial universal (.mtxu)")
    ventana.geometry("560x380")

    estado = {"ruta_original": None, "ruta_comprimida": None}

    lbl_archivo = tk.Label(ventana, text="Ningún archivo seleccionado", wraplength=520)
    lbl_modo = tk.Label(ventana, text="Modo: -")
    lbl_tam_original = tk.Label(ventana, text="Tamaño original: -")
    lbl_tam_comprimido = tk.Label(ventana, text="Tamaño comprimido: -")
    lbl_reduccion = tk.Label(ventana, text="Reducción: -")
    lbl_entropia = tk.Label(ventana, text="", wraplength=520)
    lbl_verificacion = tk.Label(ventana, text="Reconstrucción: -", font=("TkDefaultFont", 10, "bold"))

    def limpiar():
        for lbl, txt in [(lbl_modo, "Modo: -"), (lbl_tam_original, "Tamaño original: -"),
                          (lbl_tam_comprimido, "Tamaño comprimido: -"), (lbl_reduccion, "Reducción: -"),
                          (lbl_entropia, ""), (lbl_verificacion, "Reconstrucción: -")]:
            lbl.config(text=txt, fg="black")

    def seleccionar_archivo():
        ruta = filedialog.askopenfilename(filetypes=[("Todos los archivos", "*.*")])
        if ruta:
            estado["ruta_original"] = ruta
            estado["ruta_comprimida"] = None
            lbl_archivo.config(text=f"Archivo: {ruta}")
            limpiar()

    def abrir_mtxu():
        ruta = filedialog.askopenfilename(filetypes=[("Archivos .mtxu", "*.mtxu")])
        if ruta:
            estado["ruta_comprimida"] = ruta
            estado["ruta_original"] = None
            lbl_archivo.config(text=f"Archivo .mtxu: {ruta}  (listo para descomprimir)")
            limpiar()

    def comprimir():
        if not estado["ruta_original"]:
            messagebox.showwarning("Aviso", "Primero selecciona un archivo")
            return
        ventana.config(cursor="watch"); ventana.update()
        try:
            ruta_salida = estado["ruta_original"] + EXT_COMPRIMIDO
            info = comprimir_archivo(estado["ruta_original"], ruta_salida)
            estado["ruta_comprimida"] = ruta_salida
            tam_o, tam_c = info["tam_original"], info["tam_comprimido"]
            reduccion = (1 - tam_c / tam_o) * 100 if tam_o else 0
            lbl_modo.config(text=f"Modo: {info['modo_imagen']}")
            lbl_tam_original.config(text=f"Tamaño original: {tam_o} bytes")
            lbl_tam_comprimido.config(text=f"Tamaño comprimido: {tam_c} bytes")
            if reduccion < 0:
                lbl_reduccion.config(text=f"Cambio de tamaño: {reduccion:+.1f}% (CRECIÓ)")
            else:
                lbl_reduccion.config(text=f"Reducción: {reduccion:.1f}%")
            if "entropia_sin_transformar" in info:
                lbl_entropia.config(text=f"Entropía sin transformar: {info['entropia_sin_transformar']:.3f} bits/byte  |  "
                                          f"con transformación 2D: {info['entropia_con_transformar']:.3f} bits/byte")
            messagebox.showinfo("Listo", f"Archivo comprimido en:\n{ruta_salida}")
        finally:
            ventana.config(cursor="")

    def descomprimir():
        if not estado["ruta_comprimida"]:
            messagebox.showwarning("Aviso", "Primero comprime un archivo, o usa 'Abrir .mtxu'")
            return
        ventana.config(cursor="watch"); ventana.update()
        try:
            if estado["ruta_original"]:
                if "." in estado["ruta_original"].rsplit("/", 1)[-1]:
                    base, ext = estado["ruta_original"].rsplit(".", 1)
                    ruta_salida = base + "_reconstruido." + ext
                else:
                    ruta_salida = estado["ruta_original"] + "_reconstruido"
            else:
                base = estado["ruta_comprimida"]
                if base.lower().endswith(EXT_COMPRIMIDO):
                    base = base[:-len(EXT_COMPRIMIDO)]
                if "." in base.rsplit("/", 1)[-1]:
                    base2, ext = base.rsplit(".", 1)
                    ruta_salida = base2 + "_reconstruido." + ext
                else:
                    ruta_salida = base + "_reconstruido"

            descomprimir_archivo(estado["ruta_comprimida"], ruta_salida)

            if estado["ruta_original"]:
                identico = core.archivos_son_identicos(estado["ruta_original"], ruta_salida)
                if identico:
                    lbl_verificacion.config(text="RECONSTRUCCIÓN CORRECTA", fg="green")
                else:
                    lbl_verificacion.config(text="ERROR: LOS ARCHIVOS NO COINCIDEN", fg="red")
            else:
                lbl_verificacion.config(text="Archivo reconstruido (sin original en sesión para comparar)", fg="black")
            messagebox.showinfo("Listo", f"Archivo reconstruido en:\n{ruta_salida}")
        finally:
            ventana.config(cursor="")

    frame_abrir = tk.Frame(ventana); frame_abrir.pack(pady=6)
    tk.Button(frame_abrir, text="Seleccionar archivo", command=seleccionar_archivo).grid(row=0, column=0, padx=4)
    tk.Button(frame_abrir, text="Abrir .mtxu (para descomprimir)", command=abrir_mtxu).grid(row=0, column=1, padx=4)
    lbl_archivo.pack(pady=4)
    frame_botones = tk.Frame(ventana); frame_botones.pack(pady=4)
    tk.Button(frame_botones, text="Comprimir", command=comprimir).grid(row=0, column=0, padx=4)
    tk.Button(frame_botones, text="Descomprimir", command=descomprimir).grid(row=0, column=1, padx=4)
    lbl_modo.pack(pady=2)
    lbl_tam_original.pack(pady=2)
    lbl_tam_comprimido.pack(pady=2)
    lbl_reduccion.pack(pady=2)
    lbl_entropia.pack(pady=4)
    lbl_verificacion.pack(pady=8)

    ventana.mainloop()


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        print("Usa test_compresor_universal.py para la suite de pruebas.")
    else:
        iniciar_gui()
