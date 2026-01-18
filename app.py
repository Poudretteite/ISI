# noinspection PyUnresolvedReferences,PyPep8Naming
from shiny import App, render, ui, reactive
import pandas as pd
import logic

# --- UI ---
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h3("Projekt MLP"),
        # Checkbox dla TRENINGU
        ui.input_checkbox("no_header", "Plik treningowy bez naglowkow"),
        ui.input_file("file_upload", "1. Wgraj CSV (Trening)", accept=[".csv"], multiple=False),

        ui.hr(),
        ui.h5("Konfiguracja Danych"),
        ui.input_select("target_col", "Kolumna Celu (Y)", choices=[]),
        ui.input_selectize("feature_cols", "Cechy do treningu (X)", choices=[], multiple=True),

        ui.hr(),
        ui.accordion(
            ui.accordion_panel("Ustawienia Eksperymentu",
                               ui.input_select("eval_method", "Metoda Ewaluacji",
                                               choices={
                                                   "split": "Podzial Trening/Test (80/20)",
                                                   "cv": "Walidacja Krzyzowa (k-Fold)",
                                                   "shuffle": "Losowe Permutacje (ShuffleSplit)",
                                                   "loo": "Leave-One-Out (LOOCV)"
                                               }),
                               ui.panel_conditional(
                                   "input.eval_method === 'cv'",
                                   ui.input_slider("k_folds", "Liczba podzbiorow (k)", min=2, max=20, value=5)
                               ),
                               ui.hr(),
                               ui.input_text("hidden_layers", "Warstwy (np. 100,50)", value="10,10"),
                               ui.input_select("activation", "Aktywacja",
                                               choices={"relu": "ReLU", "tanh": "Tanh", "logistic": "Sigmoid"}),
                               ui.input_select("solver", "Algorytm (Solver)",
                                               choices={"adam": "Adam", "sgd": "SGD", "lbfgs": "LBFGS"},
                                               selected="adam"),
                               ui.input_slider("max_iter", "Epoki", min=50, max=1000, value=200),
                               ui.input_numeric("alpha", "Regularyzacja (Alpha)", value=0.0001, step=0.0001),
                               ),
            open=True
        ),

        ui.br(),
        ui.input_action_button("train_btn", "Uruchom Eksperyment", class_="btn-primary", width="200px")
    ),

    ui.page_fluid(
        ui.navset_tab(
            # ZAKŁADKA 1: DANE
            ui.nav_panel("Dane",
                         ui.card(
                             ui.card_header("Zarzadzanie Zbiorem Danych"),
                             ui.p("Uzyj Ctrl+Klik lub Shift+Klik, aby zaznaczyc wiele wierszy naraz.",
                                  style="color: gray; font-size: 0.9em;"),

                             ui.div(
                                 ui.input_action_button("exclude_btn", "Wyklucz zaznaczone",
                                                        class_="btn-warning btn-sm", width="160px"),
                                 ui.span(style="margin-left: 10px;"),
                                 ui.input_action_button("restore_btn", "Przywroc zaznaczone",
                                                        class_="btn-success btn-sm", width="160px"),
                                 ui.span(style="margin-left: 10px;"),
                                 ui.input_action_button("reset_all_btn", "Resetuj wszystkie",
                                                        class_="btn-secondary btn-sm", width="160px"),
                             ),

                             ui.hr(),
                             ui.h6(ui.output_text("data_status_text")),
                             ui.output_data_frame("data_preview")
                         )
                         ),

            # ZAKŁADKA 2: WYNIKI (TYLKO LEGENDA METRYK)
            ui.nav_panel("Wyniki Eksperymentu",
                         ui.output_ui("results_panel"),

                         ui.br(), ui.hr(),

                         # --- LEGENDA METRYK ---
                         ui.accordion(
                             ui.accordion_panel("Legenda Metryk - Jak czytac wyniki?",
                                                ui.markdown("""
                        **Dokladnosc (Accuracy):**
                        Ogólny procent poprawnych odpowiedzi modelu. (Np. 90% oznacza, że model myli się raz na 10 przypadków).

                        **Balanced Accuracy:**
                        Średnia skuteczność liczona oddzielnie dla każdej klasy. Bardzo ważna, gdy w danych jest nierównowaga (np. dużo zdrowych, mało chorych).

                        **Precyzja (Precision):**
                        Określa pewność modelu. Odpowiada na pytanie: "Gdy model twierdzi, że to jest klasa X, na ile procent ma rację?".

                        **Czulosc (Sensitivity / Recall):**
                        Określa wykrywalność. Odpowiada na pytanie: "Spośród wszystkich rzeczywistych przypadków klasy X, ile udało się wykryć?".

                        **F1-Score:**
                        Średnia harmoniczna między Precyzją a Czułością. Najlepsza pojedyncza liczba do oceny modelu, gdy zależy nam na obu tych aspektach naraz.

                        **Swoistosc (Specificity):**
                        Zdolność modelu do poprawnego odrzucania negatywnych przypadków (np. poprawne oznaczanie zdrowych ludzi jako zdrowych).

                        **Strata (Loss):**
                        Błąd matematyczny minimalizowany podczas nauki. Im bliżej zera, tym lepiej model dopasował się do danych treningowych.
                        """)
                                                ),
                             open=False
                         )
                         ),

            # ZAKŁADKA 3: PREDYKCJA
            ui.nav_panel("Predykcja",
                         ui.div(
                             ui.h4("Predykcja"),
                             ui.hr(),

                             # 1. Pojedyncza
                             ui.h5("1. Pojedynczy przypadek"),
                             ui.layout_columns(
                                 ui.output_ui("dynamic_inputs"),
                                 ui.div(
                                     ui.input_action_button("predict_single_btn", "Sprawdz wynik",
                                                            class_="btn-primary btn-sm", width="150px"),
                                     ui.br(), ui.br(),
                                     ui.h5(ui.output_text("prediction_result"))
                                 ),
                                 col_widths=(8, 4)
                             ),
                             ui.hr(),

                             # 2. Na wykluczonych
                             ui.h5("2. Na wykluczonych wierszach (Holdout)"),
                             ui.input_action_button("btn_holdout_predict", "Oblicz dla wykluczonych",
                                                    class_="btn-secondary btn-sm", width="200px"),
                             ui.output_ui("holdout_analysis_ui"),
                             ui.hr(),

                             # 3. Z pliku
                             ui.h5("3. Z zewnetrznego pliku CSV"),

                             ui.row(
                                 ui.column(6, ui.input_file("file_batch_pred", "Wgraj plik CSV", accept=[".csv"],
                                                            multiple=False)),
                                 ui.column(6,
                                           ui.br(),
                                           ui.input_checkbox("batch_no_header", "Ten plik nie ma naglowkow")
                                           )
                             ),

                             ui.input_action_button("btn_batch_predict", "Oblicz wyniki z pliku",
                                                    class_="btn-secondary btn-sm", width="200px"),

                             ui.output_ui("download_btn_ui"),
                             ui.br(),
                             ui.h6("Tabela wynikow:"),
                             ui.output_data_frame("batch_results_table")
                         )
                         ),

            # ZAKŁADKA 4: POMOC I PARAMETRY (NOWA)
            ui.nav_panel("Pomoc i Parametry",
                         ui.card(
                             ui.card_header("Opis Metod i Parametrow"),
                             ui.markdown("""
                    ### Metody Walidacji (Testowania)

                    * **Podzial Trening/Test:**
                      Najprostsza metoda. Dzieli dane na dwie czesci (np. 80% do nauki, 20% do testu). Jest szybka, ale wynik zalezy od losowego podzialu.

                    * **Walidacja Krzyzowa (k-Fold):**
                      Dzieli dane na k rownych czesci (np. 5). Model uczy sie 5 razy, za kazdym razem testujac na innej czesci. Wynik koncowy to srednia. Metoda bardzo rzetelna.

                    * **Losowe Permutacje (ShuffleSplit):**
                      Wielokrotnie losuje zbior treningowy i testowy, nie dbajac o sztywny podzial na rowne czesci. Pozwala sprawdzic stabilnosc modelu przy roznych losowaniach.

                    * **Leave-One-Out (LOOCV):**
                      Ekstremalna walidacja. Jesli masz 100 wierszy, model uczy sie 100 razy: za kazdym razem bierze 99 do nauki i testuje na tym jednym, ktorego nie widzial. Najdokladniejsza, ale bardzo wolna.

                    ---

                    ### Parametry Modelu (Siec Neuronowa)

                    * **Warstwy (Hidden Layers):**
                      Okresla budowe mozgu sieci. Np. "100,50" oznacza dwie warstwy ukryte: pierwsza ma 100 neuronow, druga 50. Wiecej warstw pozwala wykrywac trudniejsze zaleznosci, ale wydluza czas nauki.

                    * **Aktywacja (Activation):**
                      Funkcja matematyczna decydujaca, kiedy neuron "odpala". 
                      - *ReLU:* Najpopularniejsza, szybka i skuteczna.
                      - *Tanh / Logistic:* Starsze metody, przydatne w specyficznych przypadkach.

                    * **Algorytm (Solver):**
                      Metoda uzywana do korygowania bledow sieci podczas nauki.
                      - *Adam:* Uniwersalny, dziala dobrze na duzych danych.
                      - *SGD:* Precyzyjny, ale czesto wolniejszy.
                      - *LBFGS:* Bardzo szybki dla malych zbiorow danych (ale nie generuje wykresu straty).

                    * **Epoki (Max Iterations):**
                      Maksymalna liczba przejsc przez caly zbior danych podczas nauki. Jesli siec nauczy sie szybciej, przerwie wczesniej.

                    * **Regularyzacja (Alpha):**
                      Mechanizm zapobiegajacy "wkuwaniu na pamiec" (przeuczeniu). Wieksza wartosc Alpha zmusza siec do tworzenia prostszych, bardziej ogolnych regul.
                    """)
                         )
                         )
        )
    ),
    title="Projekt MLP"
)


# --- SERWER ---
# noinspection PyShadowingBuiltins,PyUnusedLocal,PyBroadException
def server(input, output, session):
    val = reactive.Value({
        "df": None, "excluded_indices": [],
        "model": None, "metrics": None, "y_true_eval": None, "y_pred_eval": None,
        "preprocessor": None, "features_meta": None,
        "batch_results": None, "model_blob": None,
        "holdout_metrics": None, "holdout_y_true": None, "holdout_y_pred": None
    })

    def safe_notify(msg, msg_type="default"):
        if not msg or str(msg).strip() == "": msg = "Operacja zakonczona."
        safe_type = msg_type if msg_type in ["default", "message", "warning", "error"] else "default"
        ui.notification_show(str(msg), type=safe_type)

    # --- 1. WCZYTYWANIE ---
    @reactive.Effect
    @reactive.event(input.file_upload, input.no_header)
    def load_dataset():
        file = input.file_upload()
        if not file: return
        df = logic.load_data(file[0]["datapath"], no_header=input.no_header())
        if df is not None:
            cols = df.columns.tolist()
            cat_targets = logic.get_categorical_targets(df)
            ui.update_select("target_col", choices=cat_targets)
            ui.update_selectize("feature_cols", choices=cols, selected=cols)
            val.set({
                "df": df, "excluded_indices": [],
                "model": None, "metrics": None, "y_true_eval": None, "y_pred_eval": None,
                "preprocessor": None, "features_meta": None,
                "batch_results": None, "model_blob": None,
                "holdout_metrics": None, "holdout_y_true": None, "holdout_y_pred": None
            })
            safe_notify("Plik wczytany.", "default")
        else:
            safe_notify("Blad pliku!", "error")

    @reactive.Effect
    @reactive.event(input.target_col)
    def update_feature_choices():
        state = val.get()
        if state["df"] is None: return
        target = input.target_col()
        all_cols = state["df"].columns.tolist()
        feats = [c for c in all_cols if c != target]
        ui.update_selectize("feature_cols", choices=feats, selected=feats)

    # --- ZARZĄDZANIE ---
    @render.data_frame
    def data_preview():
        state = val.get()
        df = state["df"]
        excluded = state["excluded_indices"]
        if df is not None:
            df_display = df.copy()
            status_col = ["[WYKLUCZONY]" if i in excluded else "[TRENING]" for i in df_display.index]
            df_display.insert(0, "STATUS", status_col)
            return render.DataGrid(df_display, filters=True, selection_mode="rows")
        return None

    @render.text
    def data_status_text():
        state = val.get()
        if state["df"] is None: return "Brak danych."
        total = len(state["df"])
        excluded = len(state["excluded_indices"])
        return f"Razem: {total} | Trening: {total - excluded} | Wykluczone: {excluded}"

    @reactive.Effect
    @reactive.event(input.exclude_btn)
    def exclude_rows():
        state = val.get()
        if state["df"] is None: return
        selected = input.data_preview_selected_rows()
        if not selected: safe_notify("Zaznacz wiersze.", "warning"); return
        current_excluded = set(state["excluded_indices"])
        for idx in selected: current_excluded.add(int(idx))
        new_state = state.copy()
        new_state["excluded_indices"] = list(current_excluded)
        new_state["model"] = None
        val.set(new_state)
        safe_notify("Wykluczono wiersze.", "default")

    @reactive.Effect
    @reactive.event(input.restore_btn)
    def restore_rows():
        state = val.get()
        if state["df"] is None: return
        selected = input.data_preview_selected_rows()
        if not selected: safe_notify("Zaznacz wiersze.", "warning"); return
        current_excluded = set(state["excluded_indices"])
        for idx in selected:
            if int(idx) in current_excluded: current_excluded.remove(int(idx))
        new_state = state.copy()
        new_state["excluded_indices"] = list(current_excluded)
        new_state["model"] = None
        val.set(new_state)
        safe_notify("Przywrocono wiersze.", "default")

    @reactive.Effect
    @reactive.event(input.reset_all_btn)
    def reset_all_rows():
        state = val.get()
        if state["df"] is None: return
        new_state = state.copy()
        new_state["excluded_indices"] = []
        new_state["model"] = None
        val.set(new_state)
        safe_notify("Zresetowano wykluczenia.", "default")

    # --- TRENING ---
    @reactive.Effect
    @reactive.event(input.train_btn)
    def run_training():
        state = val.get()
        df_full = state["df"]
        excluded = state["excluded_indices"]
        target = input.target_col()
        selected_feats = list(input.feature_cols())

        if df_full is None: safe_notify("Brak danych!", "error"); return
        if not target: safe_notify("Wybierz cel!", "error"); return
        if not selected_feats: safe_notify("Wybierz cechy!", "error"); return

        df_train = df_full.drop(index=excluded)
        if len(df_train) == 0: safe_notify("Brak danych do treningu.", "error"); return

        curr = val.get()
        curr.update({
            "model": None, "batch_results": None, "metrics": None
        })
        val.set(curr)

        try:
            with ui.Progress(min=1, max=10) as p:
                p.set(message="Przetwarzanie danych...", value=2)
                X, y, preproc, feats_meta = logic.preprocess_data(df_train, target, selected_features=selected_feats)
                p.set(message="Eksperyment...", value=5)

                final_model, metrics, y_true, y_pred = logic.train_and_evaluate(
                    X, y, preproc,
                    input.hidden_layers(), input.activation(), input.max_iter(),
                    input.alpha(), input.solver(),
                    method=input.eval_method(),
                    k_folds=input.k_folds()
                )

                blob = logic.save_model_pipeline(final_model, preproc, feats_meta)

                val.set({
                    "df": df_full, "excluded_indices": excluded,
                    "model": final_model,
                    "metrics": metrics,
                    "y_true_eval": y_true, "y_pred_eval": y_pred,
                    "preprocessor": preproc,
                    "features_meta": feats_meta,
                    "batch_results": None, "model_blob": blob
                })
                safe_notify("Eksperyment zakonczony!", "default")
        except Exception as e:
            msg = str(e) if str(e) else "Nieznany blad."
            safe_notify(f"Blad: {msg}", "error")

    # --- WYNIKI EKSPERYMENTU ---
    @render.ui
    def results_panel():
        state = val.get()
        if state["model"] is None:
            return ui.div(ui.h4("Brak modelu"), ui.p("Skonfiguruj i uruchom eksperyment."))

        metrics = state["metrics"]
        spec = metrics.get('specificity', 0.0)
        spec_text = f"{spec:.1%}" if spec is not None else "-"

        prec = metrics.get('precision', 0.0)
        f1 = metrics.get('f1', 0.0)
        loss = metrics.get('best_loss', 0.0)

        return ui.TagList(
            ui.h5(f"Wyniki ewaluacji ({input.eval_method()})"),
            ui.br(),
            ui.layout_columns(
                ui.value_box("Dokladnosc (Acc)", f"{metrics['accuracy']:.1%}", theme="text-blue"),
                ui.value_box("Balanced Acc", f"{metrics['balanced_accuracy']:.1%}", theme="text-green"),
                ui.value_box("Strata (Loss)", f"{loss:.4f}", theme="text-red"),
            ),
            ui.layout_columns(
                ui.value_box("Precyzja", f"{prec:.1%}", theme="bg-blue"),
                ui.value_box("Czulosc", f"{metrics['sensitivity']:.1%}", theme="bg-blue"),
                ui.value_box("F1-Score", f"{f1:.1%}", theme="bg-green"),
            ),
            ui.br(),
            ui.download_button("download_model_btn", "Pobierz Model (.pkl)", class_="btn-sm"),
            ui.br(), ui.br(),
            ui.layout_columns(
                ui.div(ui.h6("Macierz Pomylek"), ui.output_plot("plot_cm")),
                ui.div(ui.h6("Krzywa Straty"), ui.output_plot("plot_loss"))
            )
        )

    @render.plot
    def plot_cm():
        state = val.get()
        if state["model"] and state["y_true_eval"] is not None:
            return logic.get_confusion_matrix_plot(state["y_true_eval"], state["y_pred_eval"], state["model"].classes_)
        return None

    @render.plot
    def plot_loss():
        state = val.get()
        if state["model"]: return logic.get_loss_curve_plot(state["model"])
        return None

    # --- PREDYKCJE ---
    @render.ui
    def dynamic_inputs():
        meta = val.get()["features_meta"]
        if not meta: return None
        inputs = []
        for i, feat in enumerate(meta):
            elem_id = f"in_{i}"
            if feat["type"] == "cat":
                inputs.append(ui.input_select(elem_id, feat["name"], choices=feat["options"]))
            else:
                inputs.append(ui.input_numeric(elem_id, feat["name"], value=0))
        return ui.layout_columns(*inputs, col_widths=6)

    @render.text
    @reactive.event(input.predict_single_btn)
    def prediction_result():
        s = val.get()
        if not s["model"]: return ""
        try:
            row_data = {}
            for i, feat in enumerate(s["features_meta"]):
                val_input = getattr(input, f"in_{i}")()
                row_data[feat["name"]] = [val_input]
            df_row = pd.DataFrame(row_data)
            row_processed = s["preprocessor"].transform(df_row)
            if hasattr(row_processed, "toarray"): row_processed = row_processed.toarray()
            pred = s["model"].predict(row_processed)[0]
            return f"Wynik: {pred}"
        except Exception as e:
            return f"Blad: {e}"

    # HOLDOUT
    @reactive.Effect
    @reactive.event(input.btn_holdout_predict)
    def run_holdout_prediction():
        state = val.get()
        if state["model"] is None: safe_notify("Najpierw trenuj!", "warning"); return
        excluded = state["excluded_indices"]
        if not excluded: safe_notify("Brak wykluczonych wierszy.", "warning"); return
        try:
            df_full_holdout = state["df"].iloc[excluded].copy()
            required_cols = [f['name'] for f in state['features_meta']]
            if not all(col in df_full_holdout.columns for col in required_cols): raise ValueError("Brakuje kolumn.")

            df_for_model = df_full_holdout[required_cols].copy()
            for c in df_for_model.columns:
                if df_for_model[c].dtype == 'object':
                    try:
                        df_for_model[c] = df_for_model[c].str.replace(',', '.').astype(float)
                    except:
                        pass

            X_holdout = state["preprocessor"].transform(df_for_model)
            if hasattr(X_holdout, "toarray"): X_holdout = X_holdout.toarray()
            predictions = state["model"].predict(X_holdout)
            df_full_holdout.insert(0, 'WYNIK', predictions)

            # Liczymy metryki dla Holdout
            target = input.target_col()
            y_true_h = df_full_holdout[target].astype(str)
            h_metrics = logic.calculate_metrics(y_true_h, predictions, state["model"].classes_)

            new_val = state.copy()
            new_val['batch_results'] = df_full_holdout
            new_val['holdout_metrics'] = h_metrics
            new_val['holdout_y_true'] = y_true_h
            new_val['holdout_y_pred'] = predictions
            val.set(new_val)
            safe_notify(f"Obliczono wyniki dla {len(df_full_holdout)} wierszy.", "default")
        except Exception as e:
            safe_notify(f"Blad: {e}", "error")

    # UI HOLDOUT
    @render.ui
    def holdout_analysis_ui():
        m = val.get().get('holdout_metrics')
        if m:
            spec = m.get('specificity', 0.0)
            spec_text = f"{spec:.1%}" if spec is not None else "-"

            return ui.div(
                ui.br(), ui.h6("Jakosc predykcji (Holdout):"),
                ui.layout_columns(
                    ui.value_box("Dokladnosc", f"{m['accuracy']:.1%}", theme="text-blue"),
                    ui.value_box("Balanced Acc", f"{m['balanced_accuracy']:.1%}", theme="text-green"),
                ),
                ui.layout_columns(
                    ui.value_box("Czulosc", f"{m['sensitivity']:.1%}", theme="bg-blue"),
                    ui.value_box("Swoistosc", spec_text, theme="bg-green"),
                ),
                ui.output_plot("holdout_cm_plot")
            )
        return None

    @render.plot
    def holdout_cm_plot():
        s = val.get()
        if s["holdout_y_true"] is not None:
            return logic.get_confusion_matrix_plot(s["holdout_y_true"], s["holdout_y_pred"], s["model"].classes_)
        return None

    # BATCH PREDICTION
    @reactive.Effect
    @reactive.event(input.btn_batch_predict)
    def run_batch_prediction():
        state = val.get()
        file = input.file_batch_pred()
        if state["model"] is None: safe_notify("Najpierw trenuj!", "warning"); return
        if not file: safe_notify("Wgraj plik!", "warning"); return

        try:
            is_no_header = input.batch_no_header()
            df_new = logic.load_data(file[0]["datapath"], no_header=is_no_header)

            if df_new is None: raise Exception("Blad odczytu.")
            required_cols = [f['name'] for f in state['features_meta']]

            if is_no_header:
                df_new.columns = df_new.columns[:len(df_new.columns)]
                if len(df_new.columns) == len(required_cols):
                    df_new.columns = required_cols
            else:
                df_new.columns = df_new.columns.str.strip()

            try:
                df_X = df_new[required_cols].copy()
            except KeyError as e:
                if len(df_new.columns) >= len(required_cols):
                    raise ValueError(f"Niezgodnosc naglowkow. Wymagane: {required_cols}")
                else:
                    raise ValueError(f"Brakuje kolumn: {e}")

            for col in df_X.columns:
                if df_X[col].dtype == 'object':
                    try:
                        df_X[col] = df_X[col].str.replace(',', '.').astype(float)
                    except:
                        pass

            X_new = state["preprocessor"].transform(df_X)
            if hasattr(X_new, "toarray"): X_new = X_new.toarray()
            predictions = state["model"].predict(X_new)

            df_res = df_new.copy()
            if 'WYNIK' in df_res.columns: del df_res['WYNIK']
            df_res.insert(0, 'WYNIK', predictions)

            new_val = state.copy()
            new_val['batch_results'] = df_res
            val.set(new_val)

            safe_notify("Obliczono wyniki.", "default")
        except Exception as e:
            msg = str(e) if str(e) else "Nieznany blad pliku."
            safe_notify(f"Blad: {msg}", "error")

    @render.data_frame
    def batch_results_table():
        res = val.get().get('batch_results')
        if res is not None: return render.DataGrid(res, filters=True)
        return render.DataGrid(pd.DataFrame(columns=["Tu pojawia sie wyniki"]), filters=False)

    @render.ui
    def download_btn_ui():
        if val.get().get('batch_results') is not None:
            return ui.download_button("download_batch", "Pobierz Wyniki CSV", class_="btn-sm")
        return None

    @render.download(filename="model_mlp.pkl")
    def download_model_btn():
        blob = val.get().get("model_blob")
        return blob if blob else None

    @output
    @render.download()
    def download_batch():
        df = val.get().get('batch_results')
        if df is not None:
            def save(f): df.to_csv(f, index=False)

            return save
        return None


app = App(app_ui, server)