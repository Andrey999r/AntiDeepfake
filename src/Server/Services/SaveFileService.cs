using Microsoft.AspNetCore.Http;
using Services.Interfaces;

namespace Services
{
    public class SaveFileService : ISaveFileService
    {
        public async Task<string> Save(IFormFile file)
        {
            string uploadDirectory = "/app/shared-videos";

            if (!Directory.Exists(uploadDirectory))
            {
                Directory.CreateDirectory(uploadDirectory);
            }

            string filePath = Path.Combine(uploadDirectory, Guid.NewGuid().ToString() + Path.GetExtension(file.FileName));

            using (var fileStream = new FileStream(filePath, FileMode.Create))
            {
                await file.CopyToAsync(fileStream);
            }

            return filePath;
        }
    }
}
