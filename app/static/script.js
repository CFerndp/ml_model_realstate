// Elementos de la sección de predicción puntual
const form = document.getElementById("housing-form");
const resultBox = document.getElementById("result");
const predictedPrice = document.getElementById("predicted-price");
const errorBox = document.getElementById("error");

// Elementos de la sección de visualización
const featureSelect = document.getElementById("feature_name");
const btnCurve = document.getElementById("btn-curve");
let priceChart = null;

// ---- Utilidad: leer los valores base del formulario ----
function getBaseInput() {
  return {
    med_inc: parseFloat(document.getElementById("med_inc").value),
    house_age: parseFloat(document.getElementById("house_age").value),
    ave_rooms: parseFloat(document.getElementById("ave_rooms").value),
    ave_bedrooms: parseFloat(document.getElementById("ave_bedrooms").value),
    population: parseFloat(document.getElementById("population").value),
    ave_occup: parseFloat(document.getElementById("ave_occup").value),
    latitude: parseFloat(document.getElementById("latitude").value),
    longitude: parseFloat(document.getElementById("longitude").value),
  };
}

// ---- 1) Predicción puntual usando /predict_price ----
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorBox.classList.add("hidden");
  resultBox.classList.add("hidden");

  const baseInput = getBaseInput();

  try {
    const response = await fetch("/predict_price", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(baseInput),
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || "Error en la llamada a /predict_price");
    }

    const json = await response.json();
    predictedPrice.textContent = `Precio estimado: ${json.predicted_price_formatted}`;
    resultBox.classList.remove("hidden");
  } catch (err) {
    errorBox.textContent = err.message;
    errorBox.classList.remove("hidden");
  }
});

// ---- 2) Visualización de curva usando /feature_curve ----
btnCurve.addEventListener("click", async () => {
  errorBox.classList.add("hidden");

  const baseInput = getBaseInput();
  const minValue = parseFloat(document.getElementById("min_value").value);
  const maxValue = parseFloat(document.getElementById("max_value").value);
  const numPoints = parseInt(
    document.getElementById("num_points").value,
    10
  );

  const payload = {
    feature_name: featureSelect.value,
    base: baseInput,
    min_value: minValue,
    max_value: maxValue,
    num_points: numPoints,
  };

  try {
    const response = await fetch("/feature_curve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || "Error en la llamada a /feature_curve");
    }

    const json = await response.json();
    renderChart(json);
  } catch (err) {
    errorBox.textContent = err.message;
    errorBox.classList.remove("hidden");
  }
});

// ---- 3) Función para dibujar el gráfico con Chart.js ----
function renderChart(curveData) {
  const ctx = document.getElementById("priceChart").getContext("2d");

  const labels = curveData.x_values.map((v) => v.toFixed(2));
  // Mostramos precios en miles de $ para que sea más legible
  const data = curveData.prices.map((p) => p / 1000);

  if (priceChart) {
    priceChart.destroy();
  }

  priceChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        {
          label: `Precio (miles de $) vs ${curveData.feature_name}`,
          data: data,
          fill: false,
          tension: 0.2,
        },
      ],
    },
    options: {
      responsive: true,
      interaction: {
        mode: "index",
        intersect: false,
      },
      scales: {
        x: {
          title: {
            display: true,
            text: curveData.feature_name,
          },
        },
        y: {
          title: {
            display: true,
            text: "Precio estimado (miles de $)",
          },
        },
      },
    },
  });
}
