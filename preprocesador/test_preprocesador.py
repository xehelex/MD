from preprocesador import Preprocesador


def test_remove_url():
    texto = "Visita https://example.com para más información"

    resultado = Preprocesador._remove_url(texto)

    assert "https://example.com" not in resultado
    assert "Visita" in resultado


def test_remove_html_tags():
    texto = "<p>Hola <b>mundo</b></p> &amp; adiós"

    resultado = Preprocesador._remove_html_tags(texto)

    # Las etiquetas se cambian por espacios para no pegar palabras; los espacios sobran después
    assert resultado.split() == ["Hola", "mundo", "&", "adiós"]


def test_chat_conversion():
    tokens = ["lol", "ok"]

    chat_words = {
        "lol": "laughing out loud"
    }

    resultado = Preprocesador._chat_conversion(
        tokens,
        chat_words
    )

    assert resultado == ["laughing out loud", "ok"]


def test_negaciones_en():
    assert Preprocesador._negaciones_en("I don't like it") == "I do not like it"
    assert Preprocesador._negaciones_en("i can't and i won't") == "i can not and i will not"
    assert Preprocesador._negaciones_en("it's fine") == "it fine"
