import '@testing-library/jest-dom';
import { vi } from 'vitest';


Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: vi.fn().mockImplementation(query => ({
        matches: false,
        media: query,
        onchange: null,
        addListener: vi.fn(),
        removeListener: vi.fn(),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        dispatchEvent: vi.fn(),
    })),
});
export const mockDexieTables: Record<string, any> = {};

vi.mock('dexie', async () => {
    class MockDexieTable {
        public name: string;
        constructor(name: string) {
            this.name = name;
        }
        count = vi.fn().mockResolvedValue(0);
        add = vi.fn().mockResolvedValue(1);
        where = vi.fn().mockReturnThis();
        equals = vi.fn().mockReturnThis();
        first = vi.fn().mockResolvedValue(undefined);
        clear = vi.fn().mockResolvedValue(undefined);
    }

    class MockDexie extends (await vi.importActual<typeof DexieOriginal>('dexie')).default {
        constructor(name: string) {
            super(name, { indexedDB: vi.fn() as any, IDBKeyRange: vi.fn() as any });
            // console.log(`MockDexie created: ${name}`);
        }

        version(versionNumber: number) {
            // @ts-ignore
            return {
                stores: (schema: any) => {
                    Object.keys(schema).forEach(tableName => {
                        if (!mockDexieTables[tableName]) {
                            mockDexieTables[tableName] = new MockDexieTable(tableName);
                        }
                    });
                },
            };
        }

        table(name: string) {
            if (!mockDexieTables[name]) {
                mockDexieTables[name] = new MockDexieTable(name);
            }
            return mockDexieTables[name] as any;
        }
        open = vi.fn().mockResolvedValue(this as any);
    }
    return { default: MockDexie };
});


global.URL.createObjectURL = vi.fn(() => 'mock-object-url');
global.URL.revokeObjectURL = vi.fn();

HTMLMediaElement.prototype.play = vi.fn(() => Promise.resolve());
HTMLMediaElement.prototype.pause = vi.fn();
HTMLMediaElement.prototype.load = vi.fn();
