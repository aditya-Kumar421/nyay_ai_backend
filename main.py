from fastapi import FastAPI, Depends, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from pydantic import BaseModel, ConfigDict
from typing import List, Optional
import os

from datetime import datetime
from dotenv import load_dotenv 
import google.genai as genai


load_dotenv()  
DATABASE_URL = os.getenv(
    "DATABASE_URL"
)
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Module-level clients/warm connections (set at startup)
gemini_client = None
startup_db_conn = None


# ========================
# SQLALCHEMY MODELS
# ========================
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)
    role = Column(String, nullable=False)
    phone = Column(String(20), nullable=True)

    cases = relationship(
        "Case",
        back_populates="petitioner",
        foreign_keys="Case.user_id"
    )
    lawyer_profile = relationship("LawyerProfile", uselist=False, back_populates="user")
    availabilities = relationship("Availability", back_populates="lawyer")
    sent_messages = relationship("Message", foreign_keys="Message.sender_id", back_populates="sender")
    received_messages = relationship("Message", foreign_keys="Message.receiver_id", back_populates="receiver")


class LawyerProfile(Base):
    __tablename__ = "lawyer_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    specialization = Column(String, nullable=False)
    location = Column(String, nullable=False)
    fees = Column(Float, nullable=False)
    experience = Column(Integer, nullable=False)
    rating = Column(Float, default=4.5, nullable=False)
    court_of_practice = Column(String, nullable=True)
    bar_council_id = Column(String, nullable=True)

    user = relationship("User", back_populates="lawyer_profile")


class Case(Base):
    __tablename__ = "cases"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    description = Column(String, nullable=False)
    budget = Column(Float, nullable=False)
    location = Column(String, nullable=False)
    status = Column(String, default="pending")
    assigned_lawyer_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    petitioner = relationship(
        "User",
        back_populates="cases",
        foreign_keys=[user_id]
    )


class Review(Base):
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(Integer, ForeignKey("cases.id"), nullable=False)
    petitioner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    lawyer_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    rating = Column(Float, nullable=False)
    comment = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Availability(Base):
    __tablename__ = "availabilities"

    id = Column(Integer, primary_key=True, index=True)
    lawyer_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    date = Column(String, nullable=False)
    time_slot = Column(String, nullable=False)
    is_booked = Column(Boolean, default=False)

    lawyer = relationship("User", back_populates="availabilities")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(Integer, ForeignKey("cases.id"), nullable=False)
    sender_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    receiver_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    content = Column(String, nullable=False)
    purpose = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    is_read = Column(Boolean, default=False)

    sender = relationship("User", foreign_keys=[sender_id], back_populates="sent_messages")
    receiver = relationship("User", foreign_keys=[receiver_id], back_populates="received_messages")


class DemoRequest(Base):
    __tablename__ = "demo_requests"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, nullable=False)
    message = Column(String, nullable=False)


# ========================
# PYDANTIC SCHEMAS (unchanged)
# ========================
class UserCreate(BaseModel):
    name: str
    email: str
    password: str
    role: str
    phone: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    role: str
    phone: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class UserLogin(BaseModel):
    email: str
    password: str


class CaseCreate(BaseModel):
    user_id: int
    description: str
    budget: float
    location: str


class CaseResponse(BaseModel):
    id: int
    user_id: int
    description: str
    budget: float
    location: str
    status: str
    assigned_lawyer_id: Optional[int] = None

    model_config = ConfigDict(from_attributes=True)


class CaseStatusUpdate(BaseModel):
    status: str
    assigned_lawyer_id: Optional[int] = None


class LawyerProfileBase(BaseModel):
    specialization: str
    location: str
    fees: float
    experience: int
    court_of_practice: Optional[str] = None
    bar_council_id: Optional[str] = None


class LawyerProfileCreate(LawyerProfileBase):
    user_id: int


class LawyerProfileResponse(LawyerProfileBase):
    id: int
    user_id: int
    rating: float
    court_of_practice: Optional[str] = None
    bar_council_id: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class RecommendRequest(BaseModel):
    description: str
    budget: float
    location: str


class LawyerRecommendation(BaseModel):
    user_id: int
    id: int
    lawyer_name: str
    specialization: str
    location: str
    fees: float
    experience: int
    rating: float
    match_score: int
    reason: str
    court_of_practice: Optional[str] = None
    bar_council_id: Optional[str] = None
    
    model_config = ConfigDict(from_attributes=True)


class ReviewCreate(BaseModel):
    case_id: int
    rating: float
    comment: Optional[str] = None


class AvailabilityCreate(BaseModel):
    date: str
    time_slot: str


class AvailabilityResponse(BaseModel):
    id: int
    date: str
    time_slot: str
    is_booked: bool

    model_config = ConfigDict(from_attributes=True)


class MessageCreate(BaseModel):
    case_id: int
    receiver_id: int
    content: str
    purpose: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class MessageResponse(BaseModel):
    id: int
    sender_name: str
    receiver_name: str
    content: str
    purpose: Optional[str] = None
    timestamp: datetime
    is_read: bool

    model_config = ConfigDict(from_attributes=True)


class ReceiverRequest(BaseModel):
    receiver_id: int

    model_config = ConfigDict(from_attributes=True)


class SenderRequest(BaseModel):
    sender_id: int

    model_config = ConfigDict(from_attributes=True)


class MessageFullResponse(BaseModel):
    id: int
    case_id: int
    sender_id: int
    receiver_id: int
    content: str
    purpose: Optional[str] = None
    timestamp: datetime
    is_read: bool

    model_config = ConfigDict(from_attributes=True)


class DemoRequestCreate(BaseModel):
    name: str
    email: str
    message: str


class BulkLawyerItem(BaseModel):
    name: str
    email: str
    password: str
    phone: Optional[str] = None
    specialization: str
    location: str
    fees: float
    experience: int
    court_of_practice: Optional[str] = None
    bar_council_id: Optional[str] = None



class BulkCaseItem(BaseModel):
    user_id: int
    description: str
    budget: float
    location: str

# ========================
# DATABASE DEPENDENCY
# ========================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ========================
# HELPER FUNCTIONS
# ========================
def calculate_score(
    lawyer_profile: LawyerProfile,
    case_description: str,
    case_budget: float,
    case_location: str
) -> tuple[int, str]:
    score = 0
    reason_parts = []

    desc_lower = case_description.lower()
    spec_lower = lawyer_profile.specialization.lower()
    loc_lower = case_location.lower()

    if spec_lower in desc_lower or any(word in desc_lower for word in spec_lower.split()):
        score += 2
        reason_parts.append("Specialization match")

    if lawyer_profile.fees <= case_budget:
        score += 1
        reason_parts.append("Within budget")

    if lawyer_profile.location.lower() == loc_lower:
        score += 1
        reason_parts.append("Location match")

    reason = " + ".join(reason_parts) if reason_parts else "Basic match"
    return score, reason

def classify_case_type(description: str) -> str:
    """
    Classifies a legal case into exactly one category:
    'family', 'property', or 'criminal' using Gemini Flash.
    """
    try:
        # Reuse a startup-initialized Gemini client when available to avoid per-call init
        global gemini_client
        client = gemini_client
        if client is None:
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        prompt = f"""
                You are an expert Indian legal classifier.

                Your task is to analyze a legal case description and classify it into **exactly one** of the following three categories based on Indian law:

                - family
                - property
                - criminal

                ### Classification Rules (Indian Legal Context):
                - **family**: Any dispute related to marriage, divorce, child custody, maintenance, domestic violence, inheritance within family, adoption, or family relations.
                - **property**: Any dispute related to land, house, ownership, title, sale/purchase, partition, rent, eviction, or any civil property matter (without any criminal act like murder, assault, cheating, etc.).
                - **criminal**: Any offence involving crime under Bharatiya Nyaya Sanhita (BNS) / IPC such as murder, assault, theft, fraud, cheating, rape, FIR, police case, or any cognizable offence. Even if there is a property or family background, if a crime has been committed, it is criminal.

                ### Very Important Instructions:
                - You must return **ONLY ONE WORD** as output: either `family`, `property`, or `criminal`.
                - Do not write any explanation, reason, sentence, or extra text.
                - Do not use quotes or any formatting.
                - If the case has both civil and criminal elements, prioritize **criminal** as it is more serious under Indian law.

                Now classify the following case description:

                Case: {description}
"""

        # New way to call Gemini Flash
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        result = response.text.strip().lower()

        # Safety fallback
        if result not in ["family", "property", "criminal"]:
            return "family"

        return result

    except Exception as e:
        print("Gemini error:", e)
        return "family"  # fallback on failure

def update_lawyer_rating(db: Session, lawyer_id: int):
    reviews = db.query(Review).filter(Review.lawyer_id == lawyer_id).all()
    if reviews:
        avg_rating = sum(r.rating for r in reviews) / len(reviews)
        profile = db.query(LawyerProfile).filter(LawyerProfile.user_id == lawyer_id).first()
        if profile:
            profile.rating = round(avg_rating, 1)
            db.commit()


# Use FastAPI startup/shutdown events to initialize expensive clients and warm DB

def _seed_sample_data_if_needed():
    db = SessionLocal()
    try:
        if db.query(User).count() == 0:
            p1 = User(name="Aditya Kumar", email="aditya@example.com", password="pass123",
                      role="petitioner", phone="9876543210")
            db.add(p1)
            db.commit()
            db.refresh(p1)

            case1 = Case(user_id=p1.id, description="Family dispute and divorce in Delhi",
                         budget=8000.0, location="Delhi", status="matched", assigned_lawyer_id=3)
            db.add(case1)

            p2 = User(name="Priya Sharma", email="priya@example.com", password="pass123",
                      role="petitioner", phone="9123456789")
            db.add(p2)
            db.commit()
            db.refresh(p2)

            case2 = Case(user_id=p2.id, description="Criminal case FIR in Mumbai",
                         budget=15000.0, location="Mumbai", status="in_progress", assigned_lawyer_id=4)
            db.add(case2)

            l1 = User(name="Rajesh Gupta", email="rajesh.lawyer@example.com", password="pass123",
                      role="lawyer", phone="9988776655")
            db.add(l1)
            db.commit()
            db.refresh(l1)
            prof1 = LawyerProfile(user_id=l1.id, specialization="family", location="Delhi",
                                  fees=4500.0, experience=8, rating=4.9,
                                  court_of_practice="High Court", bar_council_id="HC-DEL-001")
            db.add(prof1)

            l2 = User(name="Meena Patel", email="meena.lawyer@example.com", password="pass123",
                      role="lawyer", phone="9871234567")
            db.add(l2)
            db.commit()
            db.refresh(l2)
            prof2 = LawyerProfile(user_id=l2.id, specialization="criminal", location="Mumbai",
                                  fees=12000.0, experience=12, rating=4.7,
                                  court_of_practice="Sessions Court", bar_council_id="SC-MUM-002")
            db.add(prof2)

            l3 = User(name="Amit Singh", email="amit.lawyer@example.com", password="pass123",
                      role="lawyer", phone="9123456780")
            db.add(l3)
            db.commit()
            db.refresh(l3)
            prof3 = LawyerProfile(user_id=l3.id, specialization="property", location="Delhi",
                                  fees=6000.0, experience=6, rating=4.5,
                                  court_of_practice="District Court", bar_council_id="DC-DEL-003")
            db.add(prof3)

            l4 = User(name="Sneha Rao", email="sneha.lawyer@example.com", password="pass123",
                      role="lawyer", phone="9988771122")
            db.add(l4)
            db.commit()
            db.refresh(l4)
            prof4 = LawyerProfile(user_id=l4.id, specialization="family", location="Mumbai",
                                  fees=3500.0, experience=4, rating=4.8,
                                  court_of_practice="Subordinate Court", bar_council_id="SUB-MUM-004")
            db.add(prof4)

            avail1 = Availability(lawyer_id=l1.id, date="2026-04-28", time_slot="10:00 AM - 11:00 AM")
            avail2 = Availability(lawyer_id=l1.id, date="2026-04-28", time_slot="02:00 PM - 03:00 PM")
            avail3 = Availability(lawyer_id=l2.id, date="2026-04-29", time_slot="11:00 AM - 12:00 PM")
            db.add_all([avail1, avail2, avail3])

            review1 = Review(case_id=1, petitioner_id=1, lawyer_id=3, rating=5.0,
                             comment="Excellent lawyer, very professional!")
            db.add(review1)

            demo = DemoRequest(name="Test User", email="test@example.com", message="I want to see a demo")
            db.add(demo)

            db.commit()
            print("✅ Nyay AI sample data inserted successfully!")
    finally:
        db.close()


# Create the FastAPI app (no lifespan manager)
app = FastAPI(
    title="Nyay AI - Legal Tech Backend",
    description="Complete FastAPI backend with Case Lifecycle, Reviews, Search, Availability & Messaging",
    version="1.1.0",
)


@app.on_event("startup")
def on_startup():
    global gemini_client, startup_db_conn
    # Initialize Gemini client
    try:
        gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        print("✅ Gemini client initialized at startup")
    except Exception as e:
        gemini_client = None
        print("⚠️ Failed to initialize Gemini at startup:", e)

    # Warm DB connection and create tables + sample data
    try:
        startup_db_conn = engine.connect()
        print("✅ DB connection opened at startup")
    except Exception as e:
        startup_db_conn = None
        print("⚠️ Failed to open DB connection at startup:", e)

    Base.metadata.create_all(bind=engine)
    _seed_sample_data_if_needed()


@app.on_event("shutdown")
def on_shutdown():
    global startup_db_conn, gemini_client
    try:
        if startup_db_conn is not None:
            startup_db_conn.close()
            print("✅ DB startup connection closed")
    except Exception:
        pass
    try:
        gemini_client = None
    except Exception:
        pass


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========================
# AUTH ENDPOINTS
# ========================
@app.post("/signup", response_model=UserResponse)
def signup(user_data: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == user_data.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    if user_data.role not in ["petitioner", "lawyer"]:
        raise HTTPException(status_code=400, detail="Role must be 'petitioner' or 'lawyer'")

    new_user = User(
        name=user_data.name,
        email=user_data.email,
        password=user_data.password,
        role=user_data.role,
        phone=user_data.phone
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user


@app.post("/login", response_model=UserResponse)
def login(login_data: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(
        User.email == login_data.email,
        User.password == login_data.password
    ).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return user


# ========================
# PETITIONER ENDPOINTS
# ========================
@app.post("/add-case", response_model=dict)
def add_case(case_data: CaseCreate, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == case_data.user_id, User.role == "petitioner").first()
    if not user:
        raise HTTPException(status_code=400, detail="Invalid petitioner ID")

    new_case = Case(
        user_id=case_data.user_id,
        description=case_data.description,
        budget=case_data.budget,
        location=case_data.location,
        status="pending"
    )
    db.add(new_case)
    db.commit()
    db.refresh(new_case)
    return {"message": "Case added successfully", "case_id": new_case.id}


@app.get("/my-cases", response_model=List[CaseResponse])
def my_cases(user_id: int, db: Session = Depends(get_db)):
    cases = db.query(Case).filter(Case.user_id == user_id).all()
    return cases


# ========================
# CASE STATUS & LIFECYCLE
# ========================
@app.patch("/cases/{case_id}/status", response_model=CaseResponse)
def update_case_status(case_id: int, update: CaseStatusUpdate, db: Session = Depends(get_db)):
    case_obj = db.query(Case).filter(Case.id == case_id).first()
    if not case_obj:
        raise HTTPException(status_code=404, detail="Case not found")

    case_obj.status = update.status
    if update.assigned_lawyer_id:
        case_obj.assigned_lawyer_id = update.assigned_lawyer_id

    db.commit()
    db.refresh(case_obj)
    return case_obj


# ========================
# RECOMMENDATION SYSTEM
# ========================
@app.post("/recommend-lawyers", response_model=List[LawyerRecommendation])
def recommend_lawyers(request: RecommendRequest, db: Session = Depends(get_db)):

    # 🔥 NEW STEP: classify first
    case_type = classify_case_type(request.description)

    lawyer_profiles = db.query(LawyerProfile).all()
    recommendations = []

    for lp in lawyer_profiles:
        lawyer_user = db.query(User).filter(User.id == lp.user_id).first()
        if not lawyer_user:
            continue

        # 👇 pass case_type instead of description
        score, reason = calculate_score(
            lp,
            case_type,
            request.budget,
            request.location
        )

        rec = LawyerRecommendation(
            id=lp.id,
            user_id=lawyer_user.id,
            lawyer_name=lawyer_user.name,
            specialization=lp.specialization,
            location=lp.location,
            fees=lp.fees,
            experience=lp.experience,
            rating=lp.rating,
            match_score=score,
            reason=reason
            ,court_of_practice=lp.court_of_practice,
            bar_council_id=lp.bar_council_id
        )
        recommendations.append(rec)

    recommendations.sort(key=lambda x: x.match_score, reverse=True)
    return recommendations[:5]


# ========================
# ADVANCED LAWYER SEARCH
# ========================
@app.get("/search-lawyers", response_model=List[LawyerRecommendation])
def search_lawyers(
    specialization: Optional[str] = Query(None),
    location: Optional[str] = Query(None),
    min_experience: Optional[int] = Query(None),
    max_fees: Optional[float] = Query(None),
    min_rating: Optional[float] = Query(0.0),
    db: Session = Depends(get_db)
):
    query = db.query(LawyerProfile)

    if specialization:
        query = query.filter(LawyerProfile.specialization.ilike(f"%{specialization}%"))
    if location:
        query = query.filter(LawyerProfile.location.ilike(f"%{location}%"))
    if min_experience is not None:
        query = query.filter(LawyerProfile.experience >= min_experience)
    if max_fees is not None:
        query = query.filter(LawyerProfile.fees <= max_fees)
    if min_rating > 0:
        query = query.filter(LawyerProfile.rating >= min_rating)

    profiles = query.all()
    results = []

    for lp in profiles:
        lawyer_user = db.query(User).filter(User.id == lp.user_id).first()
        if not lawyer_user:
            continue
        results.append(LawyerRecommendation(
            id=lp.id,
            user_id=lawyer_user.id,
            lawyer_name=lawyer_user.name,
            specialization=lp.specialization,
            location=lp.location,
            fees=lp.fees,
            experience=lp.experience,
            rating=lp.rating,
            match_score=0,
            reason="Filter match"
            ,court_of_practice=lp.court_of_practice,
            bar_council_id=lp.bar_council_id
        ))
    return results


# ========================
# LAWYER ENDPOINTS
# ========================
@app.post("/create-profile", response_model=LawyerProfileResponse)
def create_profile(profile_data: LawyerProfileCreate, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == profile_data.user_id, User.role == "lawyer").first()
    if not user:
        raise HTTPException(status_code=400, detail="Invalid lawyer ID")

    existing = db.query(LawyerProfile).filter(LawyerProfile.user_id == profile_data.user_id).first()
    if existing:
        raise HTTPException(status_code=400, detail="Profile already exists")

    new_profile = LawyerProfile(
        user_id=profile_data.user_id,
        specialization=profile_data.specialization,
        location=profile_data.location,
        fees=profile_data.fees,
        experience=profile_data.experience,
        rating=4.5
    )
    # optional fields
    if getattr(profile_data, "court_of_practice", None):
        new_profile.court_of_practice = profile_data.court_of_practice
    if getattr(profile_data, "bar_council_id", None):
        new_profile.bar_council_id = profile_data.bar_council_id
    db.add(new_profile)
    db.commit()
    db.refresh(new_profile)
    return new_profile


@app.put("/update-profile", response_model=LawyerProfileResponse)
def update_profile(user_id: int, profile_data: LawyerProfileBase, db: Session = Depends(get_db)):
    profile = db.query(LawyerProfile).filter(LawyerProfile.user_id == user_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    for key, value in profile_data.model_dump().items():
        setattr(profile, key, value)

    db.commit()
    db.refresh(profile)
    return profile


@app.get("/my-profile", response_model=LawyerProfileResponse)
def my_profile(user_id: int, db: Session = Depends(get_db)):
    profile = db.query(LawyerProfile).filter(LawyerProfile.user_id == user_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    return profile


# ========================
# LAWYER AVAILABILITY
# ========================
@app.post("/lawyer/{lawyer_id}/availability", response_model=AvailabilityResponse)
def add_availability(lawyer_id: int, slot: AvailabilityCreate, db: Session = Depends(get_db)):
    lawyer = db.query(User).filter(User.id == lawyer_id, User.role == "lawyer").first()
    if not lawyer:
        raise HTTPException(status_code=400, detail="Invalid lawyer ID")

    new_slot = Availability(
        lawyer_id=lawyer_id,
        date=slot.date,
        time_slot=slot.time_slot
    )
    db.add(new_slot)
    db.commit()
    db.refresh(new_slot)
    return new_slot


@app.get("/lawyer/{lawyer_id}/availability", response_model=List[AvailabilityResponse])
def get_availability(lawyer_id: int, db: Session = Depends(get_db)):
    slots = db.query(Availability).filter(Availability.lawyer_id == lawyer_id).all()
    return slots


# ========================
# RATING & REVIEW
# ========================
@app.post("/submit-review", response_model=dict)
def submit_review(review_data: ReviewCreate, petitioner_id: int, db: Session = Depends(get_db)):
    case_obj = db.query(Case).filter(Case.id == review_data.case_id).first()
    if not case_obj or case_obj.user_id != petitioner_id:
        raise HTTPException(status_code=400, detail="You can only review your own cases")

    if case_obj.status != "closed":
        raise HTTPException(status_code=400, detail="Review allowed only after case is closed")

    new_review = Review(
        case_id=review_data.case_id,
        petitioner_id=petitioner_id,
        lawyer_id=case_obj.assigned_lawyer_id,
        rating=review_data.rating,
        comment=review_data.comment
    )
    db.add(new_review)
    db.commit()

    update_lawyer_rating(db, case_obj.assigned_lawyer_id)

    return {"message": "Review submitted successfully. Lawyer rating updated."}


# ========================
# MESSAGING
# ========================
@app.post("/send-message", response_model=dict)
def send_message(message_data: MessageCreate, sender_id: int, db: Session = Depends(get_db)):
    case_obj = db.query(Case).filter(Case.id == message_data.case_id).first()
    if not case_obj:
        raise HTTPException(status_code=404, detail="Case not found")

    sender = db.query(User).filter(User.id == sender_id).first()
    receiver = db.query(User).filter(User.id == message_data.receiver_id).first()
    if not sender or not receiver:
        raise HTTPException(status_code=400, detail="Invalid sender or receiver")

    new_msg = Message(
        case_id=message_data.case_id,
        sender_id=sender_id,
        receiver_id=message_data.receiver_id,
        content=message_data.content,
        purpose=message_data.purpose
    )
    db.add(new_msg)
    db.commit()
    db.refresh(new_msg)

    return {"message": "Message sent successfully", "message_id": new_msg.id}


@app.get("/case/{case_id}/messages", response_model=List[MessageResponse])
def get_case_messages(case_id: int, user_id: int, db: Session = Depends(get_db)):
    messages = db.query(Message).filter(Message.case_id == case_id).order_by(Message.timestamp).all()

    result = []
    for m in messages:
        sender_name = db.query(User.name).filter(User.id == m.sender_id).scalar()
        receiver_name = db.query(User.name).filter(User.id == m.receiver_id).scalar()
        result.append(MessageResponse(
            id=m.id,
            sender_name=sender_name or "Unknown",
            receiver_name=receiver_name or "Unknown",
            content=m.content,
            purpose=m.purpose,
            timestamp=m.timestamp,
            is_read=m.is_read
        ))
    return result


@app.post("/messages/by-sender", response_model=List[MessageFullResponse])
def get_messages_by_sender(req: SenderRequest, db: Session = Depends(get_db)):
    """Return raw message rows where `sender_id` matches the provided value."""
    messages = db.query(Message).filter(Message.sender_id == req.sender_id).order_by(Message.timestamp).all()

    result: List[MessageFullResponse] = []
    for m in messages:
        result.append(MessageFullResponse(
            id=m.id,
            case_id=m.case_id,
            sender_id=m.sender_id,
            receiver_id=m.receiver_id,
            content=m.content,
            purpose=m.purpose,
            timestamp=m.timestamp,
            is_read=m.is_read
        ))
    return result


@app.post("/messages/by-receiver", response_model=List[MessageFullResponse])
def get_messages_by_receiver(req: ReceiverRequest, db: Session = Depends(get_db)):
    messages = db.query(Message).filter(Message.receiver_id == req.receiver_id).order_by(Message.timestamp).all()

    result: List[MessageFullResponse] = []
    for m in messages:
        result.append(MessageFullResponse(
            id=m.id,
            case_id=m.case_id,
            sender_id=m.sender_id,
            receiver_id=m.receiver_id,
            content=m.content,
            purpose=m.purpose,
            timestamp=m.timestamp,
            is_read=m.is_read
        ))
    return result


# ========================
# DEMO
# ========================
@app.post("/book-demo", response_model=dict)
def book_demo(demo_data: DemoRequestCreate, db: Session = Depends(get_db)):
    new_demo = DemoRequest(
        name=demo_data.name,
        email=demo_data.email,
        message=demo_data.message
    )
    db.add(new_demo)
    db.commit()
    db.refresh(new_demo)
    return {"message": "Demo request received", "demo_id": new_demo.id}


# ========================
# BONUS
# ========================
@app.get("/all-lawyers", response_model=List[LawyerRecommendation])
def list_all_lawyers(db: Session = Depends(get_db)):
    profiles = db.query(LawyerProfile).all()
    results = []
    for lp in profiles:
        lawyer_user = db.query(User).filter(User.id == lp.user_id).first()
        if not lawyer_user:
            continue
        results.append(LawyerRecommendation(
            id=lp.id,
            user_id=lawyer_user.id,
            lawyer_name=lawyer_user.name,
            specialization=lp.specialization,
            location=lp.location,
            fees=lp.fees,
            experience=lp.experience,
            rating=lp.rating,
            match_score=0,
            reason="Listed for browsing"
            ,court_of_practice=lp.court_of_practice,
            bar_council_id=lp.bar_council_id
        ))
    return results


# ========================
# NEW BULK ADD ENDPOINT
# ========================
@app.post("/bulk-add-lawyers", response_model=dict)
def bulk_add_lawyers(
    lawyers: List[BulkLawyerItem],
    db: Session = Depends(get_db)
):
    """Add multiple lawyers in one request (User + LawyerProfile)"""
    added_count = 0
    skipped = 0

    for item in lawyers:
        # Check if email already exists
        existing = db.query(User).filter(User.email == item.email).first()
        if existing:
            skipped += 1
            continue

        # Create User
        new_user = User(
            name=item.name,
            email=item.email,
            password=item.password,
            role="lawyer",
            phone=item.phone
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)

        # Create LawyerProfile
        new_profile = LawyerProfile(
            user_id=new_user.id,
            specialization=item.specialization,
            location=item.location,
            fees=item.fees,
            experience=item.experience,
            rating=4.5
        )
        if getattr(item, "court_of_practice", None):
            new_profile.court_of_practice = item.court_of_practice
        if getattr(item, "bar_council_id", None):
            new_profile.bar_council_id = item.bar_council_id
        db.add(new_profile)
        added_count += 1

    db.commit()

    return {
        "message": "Bulk insert completed",
        "added": added_count,
        "skipped": skipped,
        "total_requested": len(lawyers)
    }

# ========================
# NEW BULK ADD CASES ENDPOINT
# ========================
@app.post("/bulk-add-cases", response_model=dict)
def bulk_add_cases(
    cases: List[BulkCaseItem],
    db: Session = Depends(get_db)
):
    """Add multiple cases in one request"""
    added_count = 0
    skipped = 0

    for item in cases:
        # Verify petitioner exists
        petitioner = db.query(User).filter(
            User.id == item.user_id,
            User.role == "petitioner"
        ).first()

        if not petitioner:
            skipped += 1
            continue

        new_case = Case(
            user_id=item.user_id,
            description=item.description,
            budget=item.budget,
            location=item.location,
            status="pending"
        )
        db.add(new_case)
        added_count += 1

    db.commit()

    return {
        "message": "Bulk cases insert completed",
        "added": added_count,
        "skipped": skipped,
        "total_requested": len(cases)
    }


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "Nyay AI Backend v1.1"}