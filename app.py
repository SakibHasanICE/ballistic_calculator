import streamlit as st
import matplotlib.pyplot as plt
from utils import load_model, predict_drop_chart

st.set_page_config(layout="wide")
st.title("AI Ballistic Scope Drop Calculator 🎯")

@st.cache_resource
def setup_model():
    return load_model("finetuned_gpt2_lora")

model, tokenizer = setup_model()

# UI Sections
with st.form("ballistic_form"):
    st.header("🔢 Ammunition Inputs")
    col1, col2, col3, col4 = st.columns(4)
    caliber = col1.text_input("Caliber", "0.308 in")
    weight = col2.text_input("Weight", "175 Gr")
    length = col3.text_input("Length", "1.24 in")
    muzzle_velocity = col4.text_input("Muzzle Velocity", "2650 fps")
    bc = st.text_input("Ballistic Coefficient", "0.453")
    
    st.header("🔧 Gun Inputs")
    col5, col6, col7, col8 = st.columns(4)
    barrel_length = col5.text_input("Barrel Length", "20 in")
    sight_height = col6.text_input("Sight Height", "1.75 in")
    twist_rate = col7.text_input("Twist Rate", "1:12 in")
    zero_range = col8.text_input("Zero Range", "100 yd")

    st.header("🌍 Environmental Inputs")
    col9, col10, col11, col12 = st.columns(4)
    temperature = col9.text_input("Temperature", "59 F")
    altitude = col10.text_input("Altitude", "0 ft")
    humidity = col11.text_input("Humidity", "50%")
    pressure = col12.text_input("Pressure", "29.92 inHg")
    wind_speed = st.text_input("Wind Speed", "10 mph")
    
    submitted = st.form_submit_button("Calculate")

if submitted:
    st.info("🔍 Predicting drop chart using fine-tuned GPT-2...")
    input_data = {
        "caliber": caliber, "weight": weight, "length": length, "muzzle_velocity": muzzle_velocity,
        "ballistic_coefficient": bc, "barrel_length": barrel_length, "sight_height": sight_height,
        "twist_rate": twist_rate, "zero_range": zero_range,
        "temperature": temperature, "altitude": altitude, "humidity": humidity,
        "pressure": pressure, "wind_speed": wind_speed
    }

    output = predict_drop_chart(model, tokenizer, input_data)

    if "error" in output:
        st.error("❌ Error generating chart.")
    else:
        st.success("✅ Prediction generated!")

        import matplotlib.pyplot as plt

        # Plot drop chart
        st.subheader("📉 Scope Drop Chart")

        x_vals = [int(d["distance"]) for d in output["drop_chart"]]
        y_vals = [float(d["drop"]) for d in output["drop_chart"]]

        # Set standard figure size (e.g., width=10in, height=5in)
        fig, ax = plt.subplots(figsize=(10, 5))  # Adjust this as needed
        ax.plot(x_vals, y_vals, marker='o', linestyle='-', color='tab:blue')

        # Styling
        ax.set_title("Drop vs Distance", fontsize=18)
        ax.set_xlabel("Distance (yd)", fontsize=14)
        ax.set_ylabel("Drop (in)", fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Show plot
        st.pyplot(fig)

        # Table View
        st.subheader("📋 Raw Drop Table")
        st.dataframe(output["drop_chart"], use_container_width=True)
