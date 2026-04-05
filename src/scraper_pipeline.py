import time
import pandas as pd
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_FILE = BASE_DIR / "datos" / "raw_data" / "raw_target_data.csv"

# Utilizamos sesión indetectable de Chrome por restricciones de Glassdoor
def get_invisible_driver():
    try:
        print("Iniciando motor indetectable...")
        options = uc.ChromeOptions()
        # Versión 146 de Chrome
        driver = uc.Chrome(options=options, use_subprocess=True, version_main=146)
        return driver
    except Exception as e:
        print(f"Error crítico al iniciar el navegador: {e}")
        return None

def scrape_glassdoor_final(url, max_pages=15):
    # Validacion de inputs
    if not url.startswith("http"):
        raise ValueError("URL inválida proporcionada para el scraping.")
    if max_pages <= 0:
        raise ValueError("El número de páginas debe ser mayor a 0.")

    driver = get_invisible_driver()
    if not driver: return pd.DataFrame()

    dataset = []
    try:
        print(f"Navegando a: {url}")
        driver.get(url)
        print("\nLogueate y presiona ENTER cuando las reseñas sean claras")
        input("...")

        for page in range(max_pages):
            try:
                print(f"--- Procesando página {page + 1} ---")
                driver.execute_script("window.scrollTo(0, 800);")
                time.sleep(3)
                wait = WebDriverWait(driver, 15)
                
                containers = wait.until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, "[class*='ReviewTextContainer_container']"))
                )

                for container in containers:
                    entry = {"pros": "N/A", "cons": "N/A"}
                    try:
                        entry["pros"] = container.find_element(By.XPATH, ".//p[contains(@class, 'green')]/following-sibling::div//p[contains(@class, 'isExpanded')]").text
                    except: pass
                    try:
                        entry["cons"] = container.find_element(By.XPATH, ".//p[contains(@class, 'red')]/following-sibling::div//p[contains(@class, 'isExpanded')]").text
                    except: pass
                    dataset.append(entry)

                # Paginación
                if page < max_pages - 1:
                    next_button = None
                    # Lista de selectores conocidos para el botón 'Siguiente'
                    selectors = ['[data-test="pagination-next"]', '.nextButton', 'button[aria-label="Next"]']
                    
                    for selector in selectors:
                        try:
                            btn = driver.find_element(By.CSS_SELECTOR, selector)
                            if btn.is_displayed():
                                next_button = btn
                                break
                        except:
                            continue

                    if next_button:
                        print("Cambiando a la siguiente página...")
                        driver.execute_script("arguments[0].click();", next_button)
                        time.sleep(5)
                    else:
                        print("No se encontró botón de 'Siguiente'. Finalizando recolección de forma segura.")
                        break

            except Exception as e:
                print(f"Aviso: Fin de recolección en página {page+1} debido a: {e}")
                break

    except Exception as e:
        print(f"Error durante el proceso de scraping: {e}")
    finally:
        try:
            driver.quit()
        except:
            pass

    return pd.DataFrame(dataset)

if __name__ == "__main__":
    try:
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        URL_TARGET = "https://www.glassdoor.com.mx/Evaluaciones/Target-Evaluaciones-E194.htm" 
        df = scrape_glassdoor_final(URL_TARGET, max_pages=15)
        
        # Validación de resultado
        if df.empty or len(df.columns) < 2:
            raise ValueError("El scraping no generó datos válidos.")
            
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"Éxito: {len(df)} registros guardados en {OUTPUT_FILE}")
    except Exception as e:
        print(f"Falla en el pipeline de scraping: {e}")