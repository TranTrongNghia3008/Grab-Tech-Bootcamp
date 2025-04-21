from fastapi import APIRouter, HTTPException, File, UploadFile
import csv
import io

# define router
router = APIRouter(
    prefix='/v1',
)

@router.post('/datasets/')
async def upload_datasets(file: UploadFile = File(...)):
    # Check if CSV or not
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail='Only CSV files are supported.')

    # Read CSV file
    content = await file.read()
    decoded = content.decode('utf-8')
    reader = csv.DictReader(io.StringIO(decoded))
    records = [row for row in reader]
    
    # Save to DB
    # Not yet
    
    # Return preview
    return {'filename': file.filename, 'records': records[5]}