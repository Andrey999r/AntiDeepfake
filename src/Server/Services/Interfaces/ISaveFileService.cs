using Microsoft.AspNetCore.Http;

namespace Services.Interfaces
{
    public interface ISaveFileService
    {
        Task<string> Save(IFormFile file);
    }
}
