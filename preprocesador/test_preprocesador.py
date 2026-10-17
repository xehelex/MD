from preprocesador import Preprocesador


def test_remove_url():
    texto = "Visita https://example.com para más información"

    resultado = Preprocesador._remove_url(texto)

    assert "https://example.com" not in resultado
    assert "Visita" in resultado


def test_remove_html_tags():
    texto = "<p>Hola <b>mundo</b></p>"

    resultado = Preprocesador._remove_html_tags(texto)

    assert resultado == "Hola mundo"


def test_remove_punc():
    texto = "Hola, mundo!"

    resultado = Preprocesador._remove_punc(
        texto,
        ",!"
    )

    assert resultado == "Hola  mundo "


def test_chat_conversion():
    texto = "LOL"

    chat_words = {
        "LOL": "laughing out loud"
    }

    resultado = Preprocesador._chat_conversion(
        texto,
        chat_words
    )

    assert resultado == "laughing out loud"


def test_negaciones_en():
    texto = "I don't like it"

    resultado = Preprocesador._negaciones_en(texto)

    assert resultado == "I dont like it"