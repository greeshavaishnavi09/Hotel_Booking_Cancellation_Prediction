
# import libraries 

import streamlit as st

from Hotel_Booking_Cancellation_Prediction.pipeline.stage_06_predictionpipeline import PredictionPipelineTrainingPipeline



# PAGE CONFIGURATION  creating page layout


st.set_page_config(
    page_title="Hotel Booking Cancellation Prediction",
    page_icon="🏨",
    layout="wide"
)


# TITLE

st.title("🏨 Hotel Booking Cancellation Prediction")

st.write(
    "Enter the hotel booking details to predict "
    "whether the booking is likely to be cancelled."
)


# INPUT FORM

with st.form("booking_form"):   

    st.subheader("Booking Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        hotel = st.selectbox(
            "Hotel",
            ["Resort Hotel", "City Hotel"]
        )

        lead_time = st.number_input(
            "Lead Time",
            min_value=0,
            value=50
        )

        arrival_date_year = st.number_input(
            "Arrival Year",
            min_value=2015,
            max_value=2030,
            value=2017
        )

        arrival_date_month = st.selectbox(
            "Arrival Month",
            [
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December"
            ]
        )

        arrival_date_week_number = st.number_input(
            "Arrival Week Number",
            min_value=1,
            max_value=53,
            value=27
        )

        arrival_date_day_of_month = st.number_input(
            "Arrival Day",
            min_value=1,
            max_value=31,
            value=10
        )

    with col2:

        stays_in_weekend_nights = st.number_input(
            "Weekend Nights",
            min_value=0,
            value=1
        )

        stays_in_week_nights = st.number_input(
            "Week Nights",
            min_value=0,
            value=3
        )

        adults = st.number_input(
            "Adults",
            min_value=1,
            value=2
        )

        children = st.number_input(
            "Children",
            min_value=0.0,
            value=0.0
        )

        babies = st.number_input(
            "Babies",
            min_value=0,
            value=0
        )

        meal = st.selectbox(
            "Meal",
            ["BB", "HB", "FB", "SC", "Undefined"]
        )

        country = st.text_input(
            "Country",
            value="PRT"
        )

    with col3:

        market_segment = st.selectbox(
            "Market Segment",
            [
                "Online TA",
                "Offline TA/TO",
                "Direct",
                "Groups",
                "Corporate",
                "Complementary",
                "Aviation"
            ]
        )

        distribution_channel = st.selectbox(
            "Distribution Channel",
            [
                "TA/TO",
                "Direct",
                "Corporate",
                "GDS"
            ]
        )

        reserved_room_type = st.text_input(
            "Reserved Room Type",
            value="A"
        )

        assigned_room_type = st.text_input(
            "Assigned Room Type",
            value="A"
        )

        deposit_type = st.selectbox(
            "Deposit Type",
            [
                "No Deposit",
                "Non Refund",
                "Refundable"
            ]
        )

        customer_type = st.selectbox(
            "Customer Type",
            [
                "Transient",
                "Transient-Party",
                "Contract",
                "Group"
            ]
        )

    st.subheader("Additional Booking Information")

    col4, col5, col6 = st.columns(3)

    with col4:

        is_repeated_guest = st.selectbox(
            "Repeated Guest",
            [0, 1]
        )

        previous_cancellations = st.number_input(
            "Previous Cancellations",
            min_value=0,
            value=0
        )

        previous_bookings_not_canceled = st.number_input(
            "Previous Bookings Not Canceled",
            min_value=0,
            value=0
        )

        booking_changes = st.number_input(
            "Booking Changes",
            min_value=0,
            value=0
        )

    with col5:

        days_in_waiting_list = st.number_input(
            "Days in Waiting List",
            min_value=0,
            value=0
        )

        adr = st.number_input(
            "ADR",
            min_value=0.0,
            value=100.0
        )

        required_car_parking_spaces = st.number_input(
            "Required Car Parking Spaces",
            min_value=0,
            value=0
        )

        total_of_special_requests = st.number_input(
            "Total Special Requests",
            min_value=0,
            value=1
        )

    with col6:

        agent = st.number_input(
            "Agent",
            min_value=0.0,
            value=9.0
        )

        company = st.number_input(
            "Company",
            min_value=0.0,
            value=0.0
        )

    submitted = st.form_submit_button(
        "🔮 Predict Cancellation"
    )


# PREDICTION


if submitted:

    # Feature engineering from data transformation
    total_nights = (
        stays_in_weekend_nights
        + stays_in_week_nights
    )

    total_guests = (
        adults
        + children
        + babies
    )

    # Create input dictionary

    input_data = {

        "hotel": hotel,
        "lead_time": lead_time,
        "arrival_date_year": arrival_date_year,
        "arrival_date_month": arrival_date_month,
        "arrival_date_week_number": arrival_date_week_number,
        "arrival_date_day_of_month": arrival_date_day_of_month,
        "stays_in_weekend_nights": stays_in_weekend_nights,
        "stays_in_week_nights": stays_in_week_nights,
        "adults": adults,
        "children": children,
        "babies": babies,
        "meal": meal,
        "country": country,
        "market_segment": market_segment,
        "distribution_channel": distribution_channel,
        "reserved_room_type": reserved_room_type,
        "assigned_room_type": assigned_room_type,
        "deposit_type": deposit_type,
        "agent": agent,
        "company": company,
        "customer_type": customer_type,
        "is_repeated_guest": is_repeated_guest,
        "previous_cancellations": previous_cancellations,
        "previous_bookings_not_canceled": previous_bookings_not_canceled,
        "booking_changes": booking_changes,
        "days_in_waiting_list": days_in_waiting_list,
        "adr": adr,
        "required_car_parking_spaces": required_car_parking_spaces,
        "total_of_special_requests": total_of_special_requests,

        # Engineered features
        "total_nights": total_nights,
        "total_guests": total_guests
    }

    # Run prediction pipeline

    try:

        pipeline = PredictionPipelineTrainingPipeline()

        prediction, probability = pipeline.main(
            input_data
        )

        st.subheader("Prediction Result")

        if prediction == 1:

            st.error(
                "⚠️ Booking is likely to be CANCELLED"
            )

        else:

            st.success(
                "✅ Booking is likely to NOT be CANCELLED"
            )

        st.write(
            f"Cancellation Probability: "
            f"{probability:.2%}"
        )

    except Exception as e:

        st.error(
            f"Prediction failed: {e}"
        )