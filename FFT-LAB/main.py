import customtkinter as ctk
from ui.main_window import MainWindow

def main():
    # Define o tema base da aplicação (Dark Mode por padrão)
    ctk.set_appearance_mode("dark")
    
    # Define a paleta de cores padrão dos botões (podemos customizar depois)
    ctk.set_default_color_theme("blue") 

    # Instancia e roda a aplicação
    app = MainWindow()
    app.mainloop()

if __name__ == "__main__":
    main()