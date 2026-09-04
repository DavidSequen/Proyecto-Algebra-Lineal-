"""
compresor_texto.py — Compresor matricial de archivos de texto (.mtxc)

Este es el programa original del proyecto: matrices candidatas en
Z/256Z (D1, D1², R, combinaciones), selección objetiva por entropía de
Shannon, y codificación de Huffman. Ahora usa el motor común
(mtxc_core.py), compartido también con compresor_universal.py y
compresor_multiarchivo.py — misma matemática, mismo formato de
registro comprimido, tres programas separados.

Ver explicacion_teorica.md y procedimientos_manuales.md para el
desarrollo matemático completo.
"""

import struct
import numpy as np
import mtxc_core as core

MAGIC = b"MTXC"
VERSION = 3
EXT_COMPRIMIDO = ".mtxc"


# ═══════════════════════════════════════════════════════════════════
# COMPRIMIR / DESCOMPRIMIR ARCHIVO (envoltorio delgado sobre el motor)
# ═══════════════════════════════════════════════════════════════════
def comprimir_archivo(ruta_entrada: str, ruta_salida: str, n: int = core.BLOCK_SIZE):
    with open(ruta_entrada, "rb") as f:
        datos = f.read()

    resultado = core.comprimir_bytes(datos, n)

    with open(ruta_salida, "wb") as f:
        f.write(MAGIC + struct.pack("<B", VERSION))
        f.write(resultado["registro"])

    resultado["tam_comprimido"] = 5 + len(resultado["registro"])
    return resultado


def descomprimir_archivo(ruta_comprimida: str, ruta_salida: str):
    with open(ruta_comprimida, "rb") as f:
        contenido = f.read()
    if contenido[:4] != MAGIC:
        raise ValueError("El archivo no tiene la firma .mtxc esperada")
    datos = core.descomprimir_bytes(contenido[5:])
    with open(ruta_salida, "wb") as f:
        f.write(datos)
    return datos


# ═══════════════════════════════════════════════════════════════════
# DEMO PARA LA EXPOSICIÓN (N=4, a mano)
# ═══════════════════════════════════════════════════════════════════
def demo_consola(n: int = 4):
    catalogo = core.catalogo_matrices(n)
    print(f"═══ DEMOSTRACIÓN CON N={n} (aritmética mod {core.MOD}) ═══\n")
    x = np.array([65, 65, 65, 66][:n], dtype=np.uint8)
    print("Vector original x:", x, " (representa texto, ej. 'AAAB')\n")

    for ids in ([1], [3], [1, 3]):
        nombre = core.nombre_pipeline(ids, catalogo)
        print(f"--- Pipeline: {nombre} ---")
        A_total = np.eye(n, dtype=np.uint8)
        for i in ids:
            A_total = catalogo[i][1] @ A_total
        print("Matriz total A:\n", A_total)
        y = core.aplicar_pipeline(x.reshape(n, 1), ids, catalogo).reshape(-1)
        print("Y = A·x =", y)
        x_reconstruido = core.deshacer_pipeline(y.reshape(n, 1), ids, catalogo).reshape(-1)
        print("X = A⁻¹·Y =", x_reconstruido)
        print(f"¿A⁻¹A == Identidad?  {core.verificar_inversa_pipeline(ids, n)}")
        print(f"¿Reconstrucción exacta?  {np.array_equal(x, x_reconstruido)}\n")


# ═══════════════════════════════════════════════════════════════════
# SUITE DE PRUEBAS (7 casos)
# ═══════════════════════════════════════════════════════════════════
def ejecutar_pruebas(n: int = core.BLOCK_SIZE):
    import os
    import time
    casos = []
    casos.append(("1. Muy repetitivo", ("abababab" * 5000).encode("utf-8")))
    casos.append(("2. Texto normal", (
        "El algebra lineal permite representar y transformar datos mediante "
        "matrices y vectores de forma reversible y verificable.\n" * 60
    ).encode("utf-8")))
    parrafo = ("Informe academico sobre transformaciones matriciales aplicadas "
               "a la compresion de archivos de texto usando algebra lineal. ")
    casos.append(("3. Texto grande (~5MB)", (parrafo * (5_000_000 // len(parrafo) + 1))[:5_000_000].encode("utf-8")))
    casos.append(("4. Caracteres UTF-8", (
        "La compresión matricial en álgebra lineal: ñ, á, é, í, ó, ú, ü, ¿?, ¡!\n" * 40
    ).encode("utf-8")))
    casos.append(("5. Archivo vacío", b""))
    casos.append(("6. No múltiplo de bloque", b"Hola mundo!!!"))
    casos.append(("7. Saltos de línea/especiales", (
        "Línea 1\r\nLínea 2\nLínea 3\tTab\t\"comillas\"\t'apóstrofe'\t%&#@!\n" * 30
    ).encode("utf-8")))

    print(f"{'Caso':32} {'Original':>10} {'Comprim.':>10} {'Reduc.':>10} {'Comp(s)':>8} {'Descomp(s)':>10}  Íntegro")
    print("-" * 100)
    os.makedirs("_pruebas_tmp", exist_ok=True)
    for nombre, datos in casos:
        ruta_in = f"_pruebas_tmp/{nombre[:2].strip('. ')}_in.txt"
        ruta_c, ruta_out = ruta_in + EXT_COMPRIMIDO, ruta_in + ".out.txt"
        with open(ruta_in, "wb") as f:
            f.write(datos)
        t0 = time.time(); info = comprimir_archivo(ruta_in, ruta_c, n); t1 = time.time()
        descomprimir_archivo(ruta_c, ruta_out); t2 = time.time()
        integro = core.archivos_son_identicos(ruta_in, ruta_out)
        tam_o, tam_c = info["tam_original"], info["tam_comprimido"]
        texto_red = f"{(1 - tam_c/tam_o)*100:.1f}%" if tam_o else "N/A"
        print(f"{nombre:32} {tam_o:>10} {tam_c:>10} {texto_red:>10} {t1-t0:>8.3f} {t2-t1:>10.3f}  {'OK' if integro else 'ERROR'}")


# ═══════════════════════════════════════════════════════════════════
# GUI
# ═══════════════════════════════════════════════════════════════════
def iniciar_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext

    ventana = tk.Tk()
    ventana.title("Compresor matricial de texto (.mtxc)")
    ventana.geometry("520x380")
    estado = {"ruta_original": None, "ruta_comprimida": None}

    lbl_archivo = tk.Label(ventana, text="Ningún archivo seleccionado", wraplength=480)
    lbl_tam_original = tk.Label(ventana, text="Tamaño original: -")
    lbl_tam_comprimido = tk.Label(ventana, text="Tamaño comprimido: -")
    lbl_reduccion = tk.Label(ventana, text="Reducción: -")
    lbl_pipeline = tk.Label(ventana, text="Matrices usadas: -", wraplength=480)
    lbl_verificacion = tk.Label(ventana, text="Reconstrucción: -", font=("TkDefaultFont", 10, "bold"))

    def limpiar():
        for lbl, txt in [(lbl_tam_original, "Tamaño original: -"), (lbl_tam_comprimido, "Tamaño comprimido: -"),
                          (lbl_reduccion, "Reducción: -"), (lbl_pipeline, "Matrices usadas: -"),
                          (lbl_verificacion, "Reconstrucción: -")]:
            lbl.config(text=txt, fg="black")

    def seleccionar_archivo():
        ruta = filedialog.askopenfilename(filetypes=[("Archivos de texto", "*.txt")])
        if ruta:
            estado["ruta_original"] = ruta; estado["ruta_comprimida"] = None
            lbl_archivo.config(text=f"Archivo: {ruta}"); limpiar()

    def abrir_mtxc():
        ruta = filedialog.askopenfilename(filetypes=[("Archivos .mtxc", "*.mtxc")])
        if ruta:
            estado["ruta_comprimida"] = ruta; estado["ruta_original"] = None
            lbl_archivo.config(text=f"Archivo .mtxc: {ruta}  (listo para descomprimir)"); limpiar()

    def comprimir():
        if not estado["ruta_original"]:
            messagebox.showwarning("Aviso", "Primero selecciona un archivo .txt"); return
        ventana.config(cursor="watch"); ventana.update()
        try:
            ruta_salida = estado["ruta_original"] + EXT_COMPRIMIDO
            info = comprimir_archivo(estado["ruta_original"], ruta_salida)
            estado["ruta_comprimida"] = ruta_salida
            tam_o, tam_c = info["tam_original"], info["tam_comprimido"]
            reduccion = (1 - tam_c / tam_o) * 100 if tam_o else 0
            lbl_tam_original.config(text=f"Tamaño original: {tam_o} bytes")
            lbl_tam_comprimido.config(text=f"Tamaño comprimido: {tam_c} bytes")
            lbl_reduccion.config(text=(f"Cambio: {reduccion:+.1f}% (CRECIÓ)" if reduccion < 0 else f"Reducción: {reduccion:.1f}%"))
            lbl_pipeline.config(text=f"Matrices usadas: {info['nombre_pipeline']}  [modo: {info['modo']}]")
            messagebox.showinfo("Listo", f"Archivo comprimido en:\n{ruta_salida}")
        finally:
            ventana.config(cursor="")

    def descomprimir():
        if not estado["ruta_comprimida"]:
            messagebox.showwarning("Aviso", "Primero comprime un archivo, o usa 'Abrir .mtxc'"); return
        ventana.config(cursor="watch"); ventana.update()
        try:
            if estado["ruta_original"]:
                ruta_salida = estado["ruta_original"].rsplit(".", 1)[0] + "_reconstruido.txt"
            else:
                base = estado["ruta_comprimida"]
                if base.lower().endswith(EXT_COMPRIMIDO):
                    base = base[:-len(EXT_COMPRIMIDO)]
                ruta_salida = (base.rsplit(".", 1)[0] if "." in base.rsplit("/", 1)[-1] else base) + "_reconstruido.txt"
            descomprimir_archivo(estado["ruta_comprimida"], ruta_salida)
            if estado["ruta_original"]:
                identico = core.archivos_son_identicos(estado["ruta_original"], ruta_salida)
                lbl_verificacion.config(text=("RECONSTRUCCIÓN CORRECTA" if identico else "ERROR: LOS ARCHIVOS NO COINCIDEN"),
                                         fg=("green" if identico else "red"))
            else:
                lbl_verificacion.config(text="Archivo reconstruido (sin original en sesión para comparar)", fg="black")
            messagebox.showinfo("Listo", f"Archivo reconstruido en:\n{ruta_salida}")
        finally:
            ventana.config(cursor="")

    def ver_demo():
        import io, contextlib
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            demo_consola(4)
        vd = tk.Toplevel(ventana); vd.title("Demo de matrices (N=4)")
        t = scrolledtext.ScrolledText(vd, width=90, height=30, font=("Courier", 9))
        t.insert("1.0", buffer.getvalue()); t.config(state="disabled"); t.pack(fill="both", expand=True)

    frame_abrir = tk.Frame(ventana); frame_abrir.pack(pady=6)
    tk.Button(frame_abrir, text="Seleccionar archivo .txt", command=seleccionar_archivo).grid(row=0, column=0, padx=4)
    tk.Button(frame_abrir, text="Abrir .mtxc (para descomprimir)", command=abrir_mtxc).grid(row=0, column=1, padx=4)
    lbl_archivo.pack(pady=4)
    frame_botones = tk.Frame(ventana); frame_botones.pack(pady=4)
    tk.Button(frame_botones, text="Comprimir", command=comprimir).grid(row=0, column=0, padx=4)
    tk.Button(frame_botones, text="Descomprimir", command=descomprimir).grid(row=0, column=1, padx=4)
    tk.Button(frame_botones, text="Ver demo de matrices (N=4)", command=ver_demo).grid(row=0, column=2, padx=4)
    lbl_tam_original.pack(pady=2); lbl_tam_comprimido.pack(pady=2); lbl_reduccion.pack(pady=2)
    lbl_pipeline.pack(pady=2); lbl_verificacion.pack(pady=8)
    ventana.mainloop()


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        demo_consola()
    elif "--test" in sys.argv:
        ejecutar_pruebas()
    else:
        iniciar_gui()
