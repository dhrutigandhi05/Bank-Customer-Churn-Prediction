from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import (
    Header,
    Footer,
    Static,
    Input,
    Select,
    Button
)
from src.predict import predict_churn

class ChurnPredictor(App):
    TITLE = "Bank Customer Churn Predictor"

    CSS = """
    Screen {
        background: $surface;
    }

    #form {
        width: 70%;
        max-width: 85;
        height: 1fr;
        padding: 1 3;
        margin: 0 1;
    }

    Input, Select {
        margin-bottom: 1;
    }

    #predict {
        margin: 1 0;
    }

    #result {
        margin-top: 1;
        padding: 1 2;
        border: round $accent;
        display: none;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header()

        with VerticalScroll(id="form"):
            yield Static("CUSTOMER INFORMATION", classes="heading")
            yield Static("Credit Score")
            yield Input(value="650", id="credit_score")
            yield Static("Geography")
            yield Select(
                [(country, country) for country in
                 ["France", "Germany", "Spain"]],
                value="France",
                id="geography"
            )

            yield Static("Gender")
            yield Select(
                [("Female", "Female"), ("Male", "Male")],
                value="Female",
                id="gender"
            )

            yield Static("Age")
            yield Input(value="45", id="age")
            yield Static("Tenure (Years)")
            yield Input(value="5", id="tenure")
            yield Static("Account Balance")
            yield Input(value="100000", id="balance")

            yield Static("Number of Products")
            yield Select(
                [(str(n), n) for n in range(1, 5)],
                value=2,
                id="products"
            )

            yield Static("Has Credit Card?")
            yield Select(
                [("Yes", 1), ("No", 0)],
                value=1,
                id="credit_card"
            )

            yield Static("Active Member?")
            yield Select(
                [("Yes", 1), ("No", 0)],
                value=0,
                id="active_member"
            )

            yield Static("Estimated Salary")
            yield Input(value="75000", id="salary")

            yield Button(
                "Predict Churn",
                id="predict",
                variant="primary"
            )

            yield Static("", id="result")

        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "predict":
            return

        try:
            customer = {
                "CreditScore": int(self.query_one("#credit_score", Input).value),
                "Geography": self.query_one("#geography", Select).value,
                "Gender": self.query_one("#gender", Select).value,
                "Age": int(self.query_one("#age", Input).value),
                "Tenure": int(self.query_one("#tenure", Input).value),
                "Balance": float(self.query_one("#balance", Input).value),
                "NumOfProducts": self.query_one("#products", Select).value,
                "HasCrCard": self.query_one("#credit_card", Select).value,
                "IsActiveMember": self.query_one("#active_member", Select).value,
                "EstimatedSalary": float(self.query_one("#salary", Input).value)
            }

            if not 18 <= customer["Age"] <= 100:
                raise ValueError("Age must be between 18 and 100.")

            if not 350 <= customer["CreditScore"] <= 850:
                raise ValueError("Credit score must be between 350 and 850.")

            if not 0 <= customer["Tenure"] <= 10:
                raise ValueError("Tenure must be between 0 and 10.")

            if customer["Balance"] < 0 or customer["EstimatedSalary"] < 0:
                raise ValueError("Balance and salary cannot be negative.")

            result = predict_churn(customer)
            probability = result["churn_probability"] * 100
            prediction = result["prediction"]

            output = (
                f"Prediction: {prediction}\n"
                f"Estimated Churn Probability: {probability:.2f}%\n\n"
                "Probability estimates are not necessarily calibrated."
            )

            result_widget = self.query_one("#result", Static)
            result_widget.update(output)
            result_widget.styles.display = "block"

        except ValueError as error:
            self.notify(str(error), severity="error")
        except Exception as error:
            self.notify(
                f"Prediction failed: {error}",
                severity="error"
            )

if __name__ == "__main__":
    ChurnPredictor().run()