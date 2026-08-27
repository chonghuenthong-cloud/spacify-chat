import streamlit as st
import time
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="BeLive Co-Living Chat",
    page_icon="🏠",
    layout="wide"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'chat_stage' not in st.session_state:
    st.session_state.chat_stage = 'initial'
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}
if 'show_area_selection' not in st.session_state:
    st.session_state.show_area_selection = False
if 'show_condo_selection' not in st.session_state:
    st.session_state.show_condo_selection = False
if 'show_form' not in st.session_state:
    st.session_state.show_form = False
if 'selected_area' not in st.session_state:
    st.session_state.selected_area = None
if 'selected_condo' not in st.session_state:
    st.session_state.selected_condo = None

# Complete property data organized by areas
AREAS_CONDOS = {
    "KL City Center": [
        "121 Residence", "7 Tree Seven", "Armani SOHO", "Austin Regency", 
        "Icon City", "Majestic Maxim", "One Cochrane", "Pixel City Central", 
        "The OOAK", "Trion KL", "Youth City"
    ],
    "Mont Kiara": [
        "Mont Kiara", "Duta Park", "M Adora", "M Vertica", 
        "The Andes", "Vertu Resort"
    ],
    "Ampang": [
        "Acacia Residence Ampang", "Astoria Ampang", "The Azure", 
        "The Azure Residences"
    ],
    "Ara Damansara": [
        "Ara Damansara", "AraTre Residence", "Emporis Kota Damansara", 
        "Kota Damansara"
    ],
    "Cheras": [
        "Arte Cheras", "Cheras", "D'Cosmos", "Razak City Residences"
    ],
    "Petaling Jaya": [
        "Parc3 Petaling Jaya", "The PARC3", "Kelana Jaya"
    ],
    "Bukit Jalil": [
        "Bora Residence Bukit Jalil", "HighPark Suites", "Platinum Splendor"
    ],
    "Setapak": [
        "Setapak", "Fairview Residence", "Keramat"
    ],
    "Subang Jaya": [
        "Subang Jaya", "Sunway Avila", "Sunway Serene"
    ],
    "Sentul": [
        "Sentul", "Vista Sentul", "Sinaran Sentul"
    ],
    "Sungai Besi": [
        "Sungai Besi", "Marina Residence", "Sapphire Paradigm"
    ],
    "Bandar Sri Permaisuri": [
        "Bandar Sri Permaisuri", "Astoria", "Epic Residence"
    ],
    "Seri Kembangan": [
        "Seri Kembangan", "Secoya Residence", "Rica Residence"
    ],
    "Others": [
        "Medini Signature", "Meta City", "MH Platinum 2", "Pinnacle", 
        "Sinaran Residence", "The Birch", "Unio Residence", 
        "Vivo Executive Apartment", "Vivo Residence"
    ]
}

# Enhanced CSS for seamless WhatsApp-like styling
st.markdown("""
<style>
/* Hide Streamlit default elements */
.stDeployButton {display: none;}
.stDecoration {display: none;}
header {visibility: hidden;}
.main .block-container {
    padding-top: 0rem;
    padding-bottom: 0rem;
    padding-left: 0rem;
    padding-right: 0rem;
    max-width: 100%;
}

/* Full page background */
.stApp {
    background-color: #e5ddd5;
}

/* Chat container styling - seamless background */
.chat-container {
    background-color: transparent;
    height: 20vh;
    overflow-y: auto;
    padding: 10px;
    margin: 0;
}

/* Message styling */
.user-message {
    background-color: #dcf8c6;
    margin: 8px 0;
    padding: 8px 12px;
    border-radius: 7px 7px 7px 7px;
    max-width: 65%;
    margin-left: auto;
    margin-right: 0;
    word-wrap: break-word;
    position: relative;
    float: right;
    clear: both;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
}

.bot-message {
    background-color: #ffffff;
    margin: 8px 0;
    padding: 8px 12px;
    border-radius: 7px 7px 7px 7px;
    max-width: 65%;
    margin-left: 0;
    margin-right: auto;
    word-wrap: break-word;
    position: relative;
    float: left;
    clear: both;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
}

.timestamp {
    font-size: 11px;
    color: #666;
    text-align: right;
    margin-top: 2px;
}

.bot-timestamp {
    font-size: 11px;
    color: #666;
    text-align: left;
    margin-top: 2px;
}

/* Header styling */
.chat-header {
    background-color: #075e54;
    color: white;
    padding: 15px;
    text-align: center;
    margin: 0;
    position: sticky;
    top: 0;
    z-index: 100;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

/* Input area styling */
.input-container {
    background-color: #f0f0f0;
    padding: 10px 20px;
    position: sticky;
    bottom: 0;
    z-index: 100;
    border-top: 1px solid #ddd;
    margin: 0;
}

/* Selection and Form styling - blend with background */
.selection-container {
    background-color: rgba(255, 255, 255, 0.9);
    padding: 15px;
    margin: 5px 20px;
    border-radius: 10px;
    border: 1px solid #ddd;
    backdrop-filter: blur(5px);
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

/* Clear floats */
.clearfix::after {
    content: "";
    display: table;
    clear: both;
}

/* Streamlit button styling */
.stButton button {
    background-color: #25d366;
    color: white;
    border: none;
    border-radius: 20px;
    padding: 8px 16px;
    margin: 2px;
    transition: background-color 0.3s;
}

.stButton button:hover {
    background-color: #128c7e;
}

/* Form elements styling */
.stSelectbox > div > div {
    background-color: white;
}

.stTextInput > div > div > input {
    background-color: white;
}

.stMultiSelect > div > div {
    background-color: white;
}

/* Radio button styling */
.stRadio > div {
    background-color: transparent;
}

/* Hide streamlit menu and footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Ensure content flows naturally */
.element-container {
    background-color: transparent !important;
}
</style>
""", unsafe_allow_html=True)

def get_current_time():
    return datetime.now().strftime("%H:%M")

def add_message(sender, content):
    st.session_state.messages.append({
        'sender': sender,
        'content': content,
        'timestamp': get_current_time()
    })

def process_user_input(user_input):
    user_input_lower = user_input.lower().strip()
    
    if st.session_state.chat_stage == 'initial':
        if any(greeting in user_input_lower for greeting in ['hi', 'hello', 'hey']):
            st.session_state.chat_stage = 'ask_area'
            st.session_state.show_area_selection = True
            return """Hello! Welcome to BeLive Co-Living! 👋

I'm here to help you find your perfect co-living space.

Which area are you interested in?"""
        else:
            return """Hello! Welcome to BeLive Co-Living! 👋

I'm here to help you find your perfect co-living space. Type 'Hi' to get started!"""
    
    elif st.session_state.chat_stage == 'form_completed':
        return handle_post_form_queries(user_input_lower)
    
    else:
        return "I'm here to help! What would you like to know about BeLive Co-Living?"

def handle_post_form_queries(user_input):
    if 'yes' in user_input.lower():
        # User confirmed budget, show room recommendations
        return f"""Great! Here are our available room recommendations for {st.session_state.selected_condo}:

🏠 **Room A102** - RM 650/month
📍 Shared bathroom, 2 housemates
🚗 Parking available

🏠 **Room B205** - RM 750/month  
📍 Private bathroom, 1 housemate
🚗 Parking available

🏠 **Room C301** - RM 680/month
📍 Shared bathroom, 3 housemates
🚗 No parking

Which room interests you? I can connect you with our agent for viewing and more details.

Reply with the room number or say "agent" to speak with our property consultant! 🏠"""
    
    elif 'no' in user_input.lower():
        return "No problem! Let me know if you'd like to explore other options or adjust your preferences."
    
    elif 'agent' in user_input.lower() or any(room in user_input.lower() for room in ['a102', 'b205', 'c301']):
        return """Perfect! I'm connecting you with our property consultant now.

📞 **Agent Contact:**
🏠 **Sarah Lim** - Senior Property Consultant
📱 WhatsApp: +60 12-345-6789
📧 Email: sarah@belive.com.my

She will contact you within 30 minutes to arrange a viewing and provide more details about the room.

Thank you for choosing BeLive Co-Living! 🎉"""
    
    else:
        return "Would you like to proceed with the available options within your budget range? Or would you like to speak with our agent directly?"

def display_area_selection():
    st.markdown("### 📍 Please select your preferred area:")
    
    areas = list(AREAS_CONDOS.keys())
    cols = st.columns(3)
    
    for i, area in enumerate(areas):
        col_idx = i % 3
        with cols[col_idx]:
            if st.button(area, key=f"area_{i}", use_container_width=True):
                st.session_state.selected_area = area
                st.session_state.show_area_selection = False
                st.session_state.show_condo_selection = True
                st.session_state.chat_stage = 'area_selected'
                
                add_message('user', area)
                add_message('bot', f"Great choice! Here are the available co-living spaces in {area}:")
                st.rerun()

def display_condo_selection():
    if st.session_state.selected_area:
        st.markdown(f"### 🏠 Available properties in {st.session_state.selected_area}:")
        
        condos = AREAS_CONDOS[st.session_state.selected_area]
        cols = st.columns(2)
        
        for i, condo in enumerate(condos):
            col_idx = i % 2
            with cols[col_idx]:
                if st.button(condo, key=f"condo_{i}", use_container_width=True):
                    st.session_state.selected_condo = condo
                    st.session_state.show_condo_selection = False
                    st.session_state.show_form = True
                    st.session_state.chat_stage = 'condo_selected'
                    
                    add_message('user', condo)
                    add_message('bot', f"Perfect! You've selected {condo}. Please fill out this form to proceed:")
                    st.rerun()

def display_inquiry_form():
    with st.form("inquiry_form"):
        st.markdown(f"### 📋 Inquiry Form - {st.session_state.selected_condo}")
        
        # 1. Room Budget/Type
        budget = st.selectbox("1. Room Budget/Type *", 
                             ["Select budget", "RM 500-700", "RM 700-900", "RM 900-1200", "RM 1200+"])
        
        # 2. How many pax staying
        pax = st.radio("2. How many pax staying? *", 
                      ["1", "2", "More than 2"])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 3. Do you have a car
            have_car = st.radio("3. Do you have a car? *", 
                               ["Yes", "No"])
            
            # 4. Need Car Park Lot
            need_parking = st.radio("4. Need Car Park Lot? *", 
                                   ["Yes", "No"])
            
            # 5. Move in date
            move_in_date = st.date_input("5. When do you plan to move in? *")
            
            # 6. Tenancy Period
            tenancy = st.radio("6. Tenancy Period *", 
                              ["6 months", "12 months"])
        
        with col2:
            # 7. Gender
            gender = st.radio("7. Gender *", 
                             ["Male", "Female"])
            
            # 8. Unit Specification (can choose multiple)
            unit_spec = st.multiselect("8. Unit Specification (can select multiple) *", 
                                      ["Female unit", "Male unit", "Mixed Gender unit"])
            
            # 10. Nationality
            nationality = st.radio("10. Nationality *", 
                                  ["Malaysia", "Others"])
            
            nationality_specify = ""
            if nationality == "Others":
                nationality_specify = st.text_input("Please specify nationality:")
        
        # 9. Workplace
        workplace = st.text_input("9. Where is your workplace? (To recommend closest property) *")
        
        submitted = st.form_submit_button("Submit Inquiry", use_container_width=True)
        
        if submitted:
            if budget != "Select budget" and pax and workplace and unit_spec:
                # Store user data
                st.session_state.user_data = {
                    'area': st.session_state.selected_area,
                    'condo': st.session_state.selected_condo,
                    'budget': budget,
                    'pax': pax,
                    'have_car': have_car,
                    'need_parking': need_parking,
                    'move_in_date': move_in_date,
                    'tenancy': tenancy,
                    'gender': gender,
                    'unit_spec': unit_spec,
                    'workplace': workplace,
                    'nationality': nationality if nationality == "Malaysia" else nationality_specify
                }
                
                st.session_state.show_form = False
                st.session_state.chat_stage = 'form_completed'
                
                # Form summary
                summary = f"""✅ **Form Summary**

📍 **Location**: {st.session_state.selected_area} - {st.session_state.selected_condo}
💰 **Budget**: {budget}
👥 **Occupants**: {pax} person(s)
🚗 **Car**: {have_car} | **Parking**: {need_parking}
📅 **Move-in**: {move_in_date}
📋 **Tenancy**: {tenancy}
👤 **Gender**: {gender}
🏠 **Unit Type**: {', '.join(unit_spec)}
🏢 **Workplace**: {workplace}
🌍 **Nationality**: {st.session_state.user_data['nationality']}

Thank you for your submission! 🎉"""
                
                add_message('bot', summary)
                
                # Check budget and send confirmation if needed
                if any(low_budget in budget for low_budget in ["RM 500-700", "RM 700-900"]):
                    confirm_message = """⚠️ **Budget Notice**

Based on your budget range, the available options might be limited. The average room price in this area is typically higher.

Would you like to proceed with options within your budget range?

Please reply "Yes" to continue or "No" to explore other alternatives."""
                    
                    add_message('bot', confirm_message)
                else:
                    # Direct to room recommendations
                    room_recs = f"""Perfect! Here are available rooms in {st.session_state.selected_condo}:

🏠 **Room A102** - RM 850/month
📍 Private bathroom, 1 housemate
🚗 Parking available

🏠 **Room B205** - RM 950/month  
📍 Private bathroom, studio-style
🚗 Parking available

🏠 **Room C301** - RM 780/month
📍 Shared bathroom, 2 housemates
🚗 No parking

Which room interests you? Reply with the room number or say "agent" to speak with our property consultant! 🏠"""
                    
                    add_message('bot', room_recs)
                
                st.rerun()
            else:
                st.error("Please fill in all required fields marked with *")

# Header
st.markdown('<div class="chat-header"><h2>🏠 BeLive Co-Living</h2><p>Customer Service Chat</p></div>', unsafe_allow_html=True)

# Main chat container
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.messages:
    if message['sender'] == 'user':
        st.markdown(f"""
        <div class="user-message">
            {message['content']}
            <div class="timestamp">{message['timestamp']}</div>
        </div>
        <div class="clearfix"></div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="bot-message">
            {message['content']}
            <div class="bot-timestamp">{message['timestamp']}</div>
        </div>
        <div class="clearfix"></div>
        """, unsafe_allow_html=True)

# Show selections and forms
if st.session_state.show_area_selection:
    st.markdown('<div class="selection-container">', unsafe_allow_html=True)
    display_area_selection()
    st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.show_condo_selection:
    st.markdown('<div class="selection-container">', unsafe_allow_html=True)
    display_condo_selection()
    st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.show_form:
    st.markdown('<div class="selection-container">', unsafe_allow_html=True)
    display_inquiry_form()
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Input area
st.markdown('<div class="input-container">', unsafe_allow_html=True)

with st.form("chat_input_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input("Type your message...", label_visibility="collapsed", placeholder="Type your message...")
    
    with col2:
        send_button = st.form_submit_button("Send", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# Handle user input
if send_button and user_input:
    # Add user message
    add_message('user', user_input)
    
    # Process and add bot response
    bot_response = process_user_input(user_input)
    add_message('bot', bot_response)
    
    # Rerun to show new messages
    st.rerun()

# Initial greeting if no messages
if not st.session_state.messages:
    initial_message = """Welcome to BeLive Co-Living! 🏠

We're excited to help you find your perfect co-living space in Kuala Lumpur.

Type 'Hi' to get started!"""
    add_message('bot', initial_message)
    st.rerun()