namespace Services.Interfaces
{
    public interface IRunModelService
    {
        Task<int> Run(string filePath);
    }
}
