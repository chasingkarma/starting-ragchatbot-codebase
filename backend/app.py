import warnings
warnings.filterwarnings("ignore", message="resource_tracker: There appear to be.*")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Any
import os

from config import config
from rag_system import RAGSystem

# Initialize FastAPI app
app = FastAPI(title="Course Materials RAG System", root_path="")

# Add trusted host middleware for proxy
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

# Enable CORS with proper settings for proxy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Initialize RAG system
rag_system = RAGSystem(config)

# Pydantic models for request/response
class QueryRequest(BaseModel):
    """Request model for course queries"""
    query: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    """Response model for course queries"""
    answer: str
    sources: List[Union[str, Dict[str, Any]]]  # Support both string and object sources
    session_id: str

class CourseStats(BaseModel):
    """Response model for course statistics"""
    total_courses: int
    course_titles: List[str]

class ClearSessionRequest(BaseModel):
    """Request model for clearing a session"""
    session_id: str

class ClearSessionResponse(BaseModel):
    """Response model for clearing a session"""
    success: bool
    message: str

class CourseOutlineRequest(BaseModel):
    """Request model for course outline"""
    course_title: str

class CourseOutlineResponse(BaseModel):
    """Response model for course outline"""
    course_title: str
    course_link: Optional[str]
    lessons: List[Dict[str, Any]]
    total_lessons: int
    formatted_outline: str

# API Endpoints

@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Process a query and return response with sources"""
    try:
        # Create session if not provided
        session_id = request.session_id
        if not session_id:
            session_id = rag_system.session_manager.create_session()
        
        # Process query using RAG system
        answer, sources = rag_system.query(request.query, session_id)
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            session_id=session_id
        )
    except Exception as e:
        print("Query error:", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/courses", response_model=CourseStats)
async def get_course_stats():
    """Get course analytics and statistics"""
    try:
        analytics = rag_system.get_course_analytics()
        return CourseStats(
            total_courses=analytics["total_courses"],
            course_titles=analytics["course_titles"]
        )
    except Exception as e:
        print("Query error:", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/clear_session", response_model=ClearSessionResponse)
async def clear_session(request: ClearSessionRequest):
    """Clear a conversation session"""
    try:
        rag_system.session_manager.clear_session(request.session_id)
        return ClearSessionResponse(
            success=True,
            message="Session cleared successfully"
        )
    except Exception as e:
        print("Clear session error:", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/course_outline", response_model=CourseOutlineResponse)
async def get_course_outline(request: CourseOutlineRequest):
    """Get course outline directly from vector store (fast)"""
    try:
        # Use the CourseOutlineTool directly for fast access
        result = rag_system.outline_tool.execute(request.course_title)

        # If it's an error message, throw HTTP exception
        if result.startswith("No course found") or result.startswith("No courses available"):
            raise HTTPException(status_code=404, detail=result)

        # Get the raw course metadata for structured response
        all_courses = rag_system.vector_store.get_all_courses_metadata()
        matching_course = None
        course_title_lower = request.course_title.lower()

        # Find matching course
        for course in all_courses:
            if course.get('title', '').lower() == course_title_lower or course_title_lower in course.get('title', '').lower():
                matching_course = course
                break

        if not matching_course:
            raise HTTPException(status_code=404, detail=f"Course not found: {request.course_title}")

        return CourseOutlineResponse(
            course_title=matching_course.get('title', 'Unknown'),
            course_link=matching_course.get('course_link'),
            lessons=matching_course.get('lessons', []),
            total_lessons=len(matching_course.get('lessons', [])),
            formatted_outline=result
        )
    except HTTPException:
        raise
    except Exception as e:
        print("Course outline error:", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Load initial documents on startup"""
    docs_path = "../docs"
    if os.path.exists(docs_path):
        print("Loading initial documents...")
        try:
            courses, chunks = rag_system.add_course_folder(docs_path, clear_existing=False)
            print(f"Loaded {courses} courses with {chunks} chunks")
        except Exception as e:
            print(f"Error loading documents: {e}")

# Custom static file handler with no-cache headers for development
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from pathlib import Path


class DevStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if isinstance(response, FileResponse):
            # Add no-cache headers for development
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response
    
    
# Serve static files for the frontend
app.mount("/", StaticFiles(directory="../frontend", html=True), name="static")