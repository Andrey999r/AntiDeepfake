using Microsoft.AspNetCore.Http;

namespace Services.Interfaces
{
    public interface ICheckFileService
    {
        Task<bool> IsVideoFileAsync(IFormFile file);
    }
}
