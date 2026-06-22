import os
import sys
import subprocess

def run_script(script_name):
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)
    print(f"\n🚀 Ejecutando: {script_name}...")
    # Ejecutar usando el mismo intérprete de python que este script
    result = subprocess.run([sys.executable, script_path], check=True)
    if result.returncode == 0:
        print(f"✅ Finalizado: {script_name}")
    else:
        print(f"❌ Error en: {script_name}")
        sys.exit(result.returncode)

if __name__ == "__main__":
    scripts = [
        "TP_F3_Preprocesamiento.py",
        "TP_F4_ModelosLineales.py",
        "TP_F5_ModelosNoLineales.py",
        "TP_F6_Optimizacion.py",
        "TP_F7_Resultados.py"
    ]
    
    for script in scripts:
        run_script(script)
        
    print("\n🎉 ¡Todos los scripts se ejecutaron con éxito y en orden!")
