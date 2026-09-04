"""
compresor_multiarchivo.py — Empaquetador matricial de varios archivos (.mtxa)

═══════════════════════════════════════════════════════════════════════
LA DECISIÓN DE DISEÑO: ¿COMPRIMIR CADA ARCHIVO POR SEPARADO, O TODOS
JUNTOS?
═══════════════════════════════════════════════════════════════════════
Hay dos estrategias razonables para empaquetar varios archivos:

  A) INDIVIDUAL: comprimir cada archivo por su cuenta (como hace ZIP
     por defecto). Cada archivo tiene su propia tabla de frecuencias
     de Huffman, ajustada exactamente a su contenido.

  B) SÓLIDO: concatenar todos los archivos en un solo bloque de bytes
     y comprimir ese bloque una sola vez (como el modo "solid" de 7z).
     Una sola tabla de frecuencias para todo, compartida entre
     archivos — mejor cuando los archivos son parecidos entre sí
     (varios .txt con vocabulario similar, por ejemplo), porque
     Huffman aprovecha una distribución de símbolos más rica.

Ninguna de las dos gana siempre: si los archivos son MUY distintos
entre sí (por ejemplo, un .txt y una imagen), mezclarlos en un solo
alfabeto diluye la distribución de frecuencias y el modo sólido puede
comprimir PEOR que el individual.

Siguiendo el mismo principio que ya usa el resto del proyecto para
elegir matrices ("medir, no adivinar"), este programa CALCULA el
resultado de ambas estrategias sobre los archivos reales que le diste,
y usa la que dé un archivo .mtxa más pequeño. La decisión (y por qué
se tomó) se le muestra al usuario, nunca se oculta.
"""

import os
import struct
import numpy as np
import mtxc_core as core

MAGIC = b"MTXA"
VERSION = 1
EXT_COMPRIMIDO = ".mtxa"


# ═══════════════════════════════════════════════════════════════════
# 1) EMPAQUETADO
# ═══════════════════════════════════════════════════════════════════
def _empaquetar_metadatos(nombres, longitudes):
    partes = [struct.pack("<H", len(nombres))]
    for nombre, longitud in zip(nombres, longitudes):
        nombre_b = nombre.encode("utf-8")
        partes.append(struct.pack("<B", len(nombre_b)) + nombre_b + struct.pack("<I", longitud))
    return b"".join(partes)


def _desempaquetar_metadatos(datos: bytes, cursor: int):
    (num_archivos,) = struct.unpack_from("<H", datos, cursor); cursor += 2
    nombres, longitudes = [], []
    for _ in range(num_archivos):
        (n,) = struct.unpack_from("<B", datos, cursor); cursor += 1
        nombre = datos[cursor:cursor + n].decode("utf-8"); cursor += n
        (longitud,) = struct.unpack_from("<I", datos, cursor); cursor += 4
        nombres.append(nombre); longitudes.append(longitud)
    return nombres, longitudes, cursor


def comprimir_multiarchivo(rutas_entrada, ruta_salida: str):
    nombres = [os.path.basename(r) for r in rutas_entrada]
    contenidos = []
    for r in rutas_entrada:
        with open(r, "rb") as f:
            contenidos.append(f.read())
    longitudes = [len(c) for c in contenidos]
    tam_original_total = sum(longitudes)

    metadatos = _empaquetar_metadatos(nombres, longitudes)

    # --- Estrategia A: individual ---
    registros_individuales = [core.comprimir_bytes(c)["registro"] for c in contenidos]
    cuerpo_individual = metadatos + b"".join(
        struct.pack("<I", len(r)) + r for r in registros_individuales
    )
    tam_individual = len(cuerpo_individual)

    # --- Estrategia B: sólido ---
    blob = b"".join(contenidos)
    resultado_solido = core.comprimir_bytes(blob)
    cuerpo_solido = metadatos + struct.pack("<I", len(resultado_solido["registro"])) + resultado_solido["registro"]
    tam_solido = len(cuerpo_solido)

    if tam_individual <= tam_solido:
        modo = 0
        cuerpo_final = cuerpo_individual
        estrategia = "individual"
    else:
        modo = 1
        cuerpo_final = cuerpo_solido
        estrategia = "sólido"

    with open(ruta_salida, "wb") as f:
        f.write(MAGIC + struct.pack("<BB", VERSION, modo) + cuerpo_final)

    tam_comprimido = 6 + len(cuerpo_final)
    return {
        "num_archivos": len(rutas_entrada),
        "nombres": nombres,
        "tam_original_total": tam_original_total,
        "tam_comprimido": tam_comprimido,
        "estrategia": estrategia,
        "tam_individual_hubiera_dado": 6 + tam_individual,
        "tam_solido_hubiera_dado": 6 + tam_solido,
    }


def descomprimir_multiarchivo(ruta_comprimida: str, carpeta_salida: str):
    with open(ruta_comprimida, "rb") as f:
        contenido = f.read()

    if contenido[:4] != MAGIC:
        raise ValueError("El archivo no tiene la firma .mtxa esperada")
    version, modo = struct.unpack("<BB", contenido[4:6])
    cuerpo = contenido[6:]

    nombres, longitudes, cursor = _desempaquetar_metadatos(cuerpo, 0)

    os.makedirs(carpeta_salida, exist_ok=True)
    rutas_generadas = []

    if modo == 0:
        for nombre, longitud in zip(nombres, longitudes):
            (n,) = struct.unpack_from("<I", cuerpo, cursor); cursor += 4
            registro = cuerpo[cursor:cursor + n]; cursor += n
            datos = core.descomprimir_bytes(registro)
            ruta = os.path.join(carpeta_salida, nombre)
            with open(ruta, "wb") as f:
                f.write(datos)
            rutas_generadas.append(ruta)
    else:
        (n,) = struct.unpack_from("<I", cuerpo, cursor); cursor += 4
        registro = cuerpo[cursor:cursor + n]; cursor += n
        blob = core.descomprimir_bytes(registro)
        pos = 0
        for nombre, longitud in zip(nombres, longitudes):
            datos = blob[pos:pos + longitud]
            pos += longitud
            ruta = os.path.join(carpeta_salida, nombre)
            with open(ruta, "wb") as f:
                f.write(datos)
            rutas_generadas.append(ruta)

    return rutas_generadas


# ═══════════════════════════════════════════════════════════════════
# 2) GUI
# ═══════════════════════════════════════════════════════════════════
def iniciar_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext

    ventana = tk.Tk()
    ventana.title("Empaquetador matricial de varios archivos (.mtxa)")
    ventana.geometry("600x460")

    estado = {"rutas_originales": [], "ruta_comprimida": None}

    lbl_archivos = scrolledtext.ScrolledText(ventana, width=70, height=8, font=("TkDefaultFont", 9))
    lbl_archivos.insert("1.0", "Ningún archivo seleccionado")
    lbl_archivos.config(state="disabled")

    lbl_estrategia = tk.Label(ventana, text="Estrategia: -")
    lbl_tam_original = tk.Label(ventana, text="Tamaño original total: -")
    lbl_tam_comprimido = tk.Label(ventana, text="Tamaño comprimido: -")
    lbl_reduccion = tk.Label(ventana, text="Reducción: -")
    lbl_comparacion = tk.Label(ventana, text="", wraplength=560, fg="#555555")
    lbl_verificacion = tk.Label(ventana, text="Reconstrucción: -", font=("TkDefaultFont", 10, "bold"))

    def mostrar_lista(rutas):
        lbl_archivos.config(state="normal")
        lbl_archivos.delete("1.0", "end")
        if rutas:
            for r in rutas:
                lbl_archivos.insert("end", r + "\n")
        else:
            lbl_archivos.insert("1.0", "Ningún archivo seleccionado")
        lbl_archivos.config(state="disabled")

    def limpiar():
        for lbl, txt in [(lbl_estrategia, "Estrategia: -"), (lbl_tam_original, "Tamaño original total: -"),
                          (lbl_tam_comprimido, "Tamaño comprimido: -"), (lbl_reduccion, "Reducción: -"),
                          (lbl_comparacion, ""), (lbl_verificacion, "Reconstrucción: -")]:
            lbl.config(text=txt, fg="black" if lbl is not lbl_comparacion else "#555555")

    def seleccionar_archivos():
        rutas = filedialog.askopenfilenames(filetypes=[("Todos los archivos", "*.*")])
        if rutas:
            estado["rutas_originales"] = list(rutas)
            estado["ruta_comprimida"] = None
            mostrar_lista(rutas)
            limpiar()

    def abrir_mtxa():
        ruta = filedialog.askopenfilename(filetypes=[("Archivos .mtxa", "*.mtxa")])
        if ruta:
            estado["ruta_comprimida"] = ruta
            estado["rutas_originales"] = []
            mostrar_lista([f"(.mtxa cargado: {ruta})"])
            limpiar()

    def comprimir():
        if not estado["rutas_originales"]:
            messagebox.showwarning("Aviso", "Primero selecciona uno o más archivos")
            return
        ventana.config(cursor="watch"); ventana.update()
        try:
            primera = estado["rutas_originales"][0]
            ruta_salida = os.path.splitext(primera)[0] + "_paquete" + EXT_COMPRIMIDO
            info = comprimir_multiarchivo(estado["rutas_originales"], ruta_salida)
            estado["ruta_comprimida"] = ruta_salida

            tam_o, tam_c = info["tam_original_total"], info["tam_comprimido"]
            reduccion = (1 - tam_c / tam_o) * 100 if tam_o else 0
            lbl_estrategia.config(text=f"Estrategia elegida: {info['estrategia']}  ({info['num_archivos']} archivos)")
            lbl_tam_original.config(text=f"Tamaño original total: {tam_o} bytes")
            lbl_tam_comprimido.config(text=f"Tamaño comprimido: {tam_c} bytes")
            if reduccion < 0:
                lbl_reduccion.config(text=f"Cambio de tamaño: {reduccion:+.1f}% (CRECIÓ)")
            else:
                lbl_reduccion.config(text=f"Reducción: {reduccion:.1f}%")
            lbl_comparacion.config(
                text=f"(individual hubiera dado {info['tam_individual_hubiera_dado']} bytes · "
                     f"sólido hubiera dado {info['tam_solido_hubiera_dado']} bytes — se usó el menor)"
            )
            messagebox.showinfo("Listo", f"Paquete creado en:\n{ruta_salida}")
        finally:
            ventana.config(cursor="")

    def descomprimir():
        if not estado["ruta_comprimida"]:
            messagebox.showwarning("Aviso", "Primero comprime archivos, o usa 'Abrir .mtxa'")
            return
        carpeta = filedialog.askdirectory(title="Elegí la carpeta donde extraer los archivos")
        if not carpeta:
            return
        ventana.config(cursor="watch"); ventana.update()
        try:
            rutas_generadas = descomprimir_multiarchivo(estado["ruta_comprimida"], carpeta)

            if estado["rutas_originales"]:
                todos_identicos = True
                detalle = []
                originales_por_nombre = {os.path.basename(r): r for r in estado["rutas_originales"]}
                for ruta_gen in rutas_generadas:
                    nombre = os.path.basename(ruta_gen)
                    original = originales_por_nombre.get(nombre)
                    if original:
                        ok = core.archivos_son_identicos(original, ruta_gen)
                        todos_identicos = todos_identicos and ok
                        detalle.append(f"{nombre}: {'OK' if ok else 'ERROR'}")
                if todos_identicos:
                    lbl_verificacion.config(text="RECONSTRUCCIÓN CORRECTA (todos los archivos)", fg="green")
                else:
                    lbl_verificacion.config(text="ERROR: " + " | ".join(detalle), fg="red")
            else:
                lbl_verificacion.config(text=f"{len(rutas_generadas)} archivo(s) extraído(s) (sin originales en sesión para comparar)", fg="black")
            messagebox.showinfo("Listo", f"Archivos extraídos en:\n{carpeta}")
        finally:
            ventana.config(cursor="")

    frame_abrir = tk.Frame(ventana); frame_abrir.pack(pady=6)
    tk.Button(frame_abrir, text="Seleccionar archivos (varios)", command=seleccionar_archivos).grid(row=0, column=0, padx=4)
    tk.Button(frame_abrir, text="Abrir .mtxa (para extraer)", command=abrir_mtxa).grid(row=0, column=1, padx=4)
    lbl_archivos.pack(pady=6, padx=12)
    frame_botones = tk.Frame(ventana); frame_botones.pack(pady=4)
    tk.Button(frame_botones, text="Comprimir", command=comprimir).grid(row=0, column=0, padx=4)
    tk.Button(frame_botones, text="Descomprimir", command=descomprimir).grid(row=0, column=1, padx=4)
    lbl_estrategia.pack(pady=2)
    lbl_tam_original.pack(pady=2)
    lbl_tam_comprimido.pack(pady=2)
    lbl_reduccion.pack(pady=2)
    lbl_comparacion.pack(pady=4)
    lbl_verificacion.pack(pady=8)

    ventana.mainloop()


if __name__ == "__main__":
    iniciar_gui()
